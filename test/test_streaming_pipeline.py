"""
Test script for streaming instrument detection pipeline.
Jobs are added to the queue at random intervals while the pipeline is consuming.
This simulates a real-time production environment.

Uses the new StreamingPipeline framework with Agent-based components.

NEEDS to be run with the docker image and GPU
"""

import ray
import time
import signal
import random
import torch
from pathlib import Path
from loguru import logger
from ray.util.queue import Queue

import hashlib
import uuid
import threading

from src.streaming_pipeline import (
    AgentRayComputeConfig,
    AgentStage,
    QueueStreamingDatasource,
    StreamingDatasourceConfig,
    StreamingPipeline,
)
from src.pipelines.instrument_detection.agents.audio_preprocessor import (
    AudioPreprocessorAgent,
)
from src.pipelines.instrument_detection.agents.instrument_detector import (
    InstrumentDetectorAgent,
    InstrumentDetectorCoTAgent,
)
from src.pipelines.instrument_detection.data_classes import InstrumentDetectJob
from src.models import load_model_and_processor

# Map string names to torch dtypes
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Default cache directory
DEFAULT_CACHE_DIR = "/app/cache"
DEFAULT_MODELS_DIR = "/app/models"


def download_model_for_vllm(
    model_name: str, models_dir: str = DEFAULT_MODELS_DIR
) -> str:
    """
    Download model files for vLLM using huggingface_hub snapshot_download.
    This downloads all model files to a local directory without loading into memory.
    Returns the path to the downloaded model.
    """
    from huggingface_hub import snapshot_download

    model_cache_name = model_name.replace("/", "_")
    local_path = Path(models_dir) / model_cache_name

    if local_path.exists() and any(local_path.iterdir()):
        logger.info(f"Model already downloaded at {local_path}")
        return str(local_path)

    logger.info(f"Downloading model {model_name} to {local_path}...")
    local_path.mkdir(parents=True, exist_ok=True)

    # Download all model files (weights, config, tokenizer, etc.)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_path),
        local_dir_use_symlinks=False,  # Copy files, don't symlink
    )

    logger.info(f"Model downloaded successfully to {local_path}")
    return str(local_path)


def cache_model(
    model_name: str, dtype: torch.dtype, cache_dir: str = DEFAULT_CACHE_DIR
) -> str:
    """
    Download and cache the model locally using save_pretrained.
    Returns the path to the cached model.
    """
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    model_cache_name = model_name.replace("/", "_")
    cache_path = Path(cache_dir) / model_cache_name

    if cache_path.exists():
        logger.info(f"Model already cached at {cache_path}")
        return str(cache_path)

    logger.info(f"Downloading and caching model {model_name} to {cache_path}")
    cache_path.mkdir(parents=True, exist_ok=True)

    # Load on CPU to avoid GPU memory issues
    logger.info("Loading model from HuggingFace...")
    model, processor = load_model_and_processor(model_name, dtype, "cpu")

    # Fix generation config conflicts before saving
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    logger.info(f"Saving model to {cache_path}")
    model.save_pretrained(cache_path)
    processor.save_pretrained(cache_path)

    # Free memory
    del model
    del processor

    logger.info(f"Model cached successfully at {cache_path}")
    return str(cache_path)


def create_job_from_file(filepath: Path) -> InstrumentDetectJob:
    """Create a job from a file path."""
    audio_bytes = filepath.read_bytes()
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    song_id = f"trk_{uuid.uuid4().hex[:12]}"
    song_hash = hashlib.sha256(audio_bytes).hexdigest()
    created_at = int(time.time())

    return InstrumentDetectJob(
        job_id=job_id,
        created_at=created_at,
        song_id=song_id,
        song_hash=song_hash,
        audio_bytes=audio_bytes,  # Pass bytes directly, not ObjectRef
        filename=filepath.name,
    )


def job_to_row(job: InstrumentDetectJob) -> dict:
    """Convert an InstrumentDetectJob to a row dict for the datasource."""
    return {
        "job_id": job.job_id,
        "song_id": job.song_id,
        "song_hash": job.song_hash,
        "filename": job.filename,
        "audio_bytes": job.audio_bytes,  # Pass bytes directly
    }


class StreamingJobProducer:
    """
    Produces jobs at random intervals to simulate real-time job submission.
    """

    def __init__(
        self,
        job_queue: Queue,
        audio_files: list[Path],
        total_jobs: int,
        min_delay_ms: int = 100,
        max_delay_ms: int = 2000,
        burst_probability: float = 0.2,
        burst_size_min: int = 3,
        burst_size_max: int = 10,
    ):
        self.job_queue = job_queue
        self.audio_files = audio_files
        self.total_jobs = total_jobs
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.burst_probability = burst_probability
        self.burst_size_min = burst_size_min
        self.burst_size_max = burst_size_max

        self.jobs_submitted = 0
        self.stop_event = threading.Event()
        self.thread = None

    def _submit_job(self) -> bool:
        """Submit a single job. Returns True if submitted."""
        if self.jobs_submitted >= self.total_jobs:
            return False

        # Pick a random audio file
        audio_file = random.choice(self.audio_files)
        job = create_job_from_file(audio_file)
        row = job_to_row(job)
        self.job_queue.put(row)
        self.jobs_submitted += 1
        return True

    def _producer_loop(self):
        """Main producer loop that runs in a separate thread."""
        logger.info(
            f"StreamingJobProducer started. Will submit {self.total_jobs} jobs."
        )

        while not self.stop_event.is_set() and self.jobs_submitted < self.total_jobs:
            # Decide if this is a burst or single submission
            if random.random() < self.burst_probability:
                # Burst submission
                burst_size = min(
                    random.randint(self.burst_size_min, self.burst_size_max),
                    self.total_jobs - self.jobs_submitted,
                )
                logger.info(f"Burst submitting {burst_size} jobs")
                for _ in range(burst_size):
                    if not self._submit_job():
                        break
            else:
                # Single submission
                self._submit_job()

            if self.jobs_submitted < self.total_jobs:
                # Random delay before next submission
                delay_ms = random.randint(self.min_delay_ms, self.max_delay_ms)
                # Use small sleep intervals to check stop_event
                sleep_until = time.time() + (delay_ms / 1000.0)
                while time.time() < sleep_until and not self.stop_event.is_set():
                    time.sleep(0.05)

            if self.jobs_submitted % 100 == 0:
                logger.info(
                    f"Producer progress: {self.jobs_submitted}/{self.total_jobs} jobs submitted"
                )

        logger.info(
            f"StreamingJobProducer finished. Submitted {self.jobs_submitted} jobs."
        )

    def start(self):
        """Start the producer thread."""
        self.thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the producer thread."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def is_done(self) -> bool:
        """Check if all jobs have been submitted."""
        return self.jobs_submitted >= self.total_jobs


def main(
    # Job producer config
    data_dir: str = "audio_files",
    total_jobs: int = 50,
    min_delay_ms: int = 100,
    max_delay_ms: int = 2000,
    burst_probability: float = 0.2,
    burst_size_min: int = 3,
    burst_size_max: int = 10,
    # Datasource config
    datasource_batch_size: int = 8,
    datasource_parallelism: int = 1,
    # Preprocessor config
    num_preprocessor_actors: int = 4,
    preprocessor_batch_size: int = 8,
    preprocessor_num_cpus: float = 1.0,
    preprocessor_max_concurrency: int = 1,
    # Detector config
    num_detector_actors: int = 1,
    detector_batch_size: int = 4,
    detector_num_gpus: float = 1.0,
    detector_max_concurrency: int = 1,
    # Model config
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    cache_dir: str = DEFAULT_CACHE_DIR,
    models_dir: str = DEFAULT_MODELS_DIR,
    skip_cache: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    # vLLM config
    use_vllm: bool = False,
    use_cot: bool = False,
    tensor_parallel_size: int = None,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 32768,
    max_num_seqs: int = 8,
):
    # Validate vLLM configuration
    if use_vllm and num_detector_actors > 1:
        logger.warning(
            f"WARNING: Using vLLM with {num_detector_actors} detector actors is NOT recommended! "
            f"Each actor will try to load the full model, causing GPU memory contention. "
            f"Use --num-detectors 1 --tensor-parallel-size {int(num_detector_actors * detector_num_gpus)} "
            f"--detector-gpus {int(num_detector_actors * detector_num_gpus)} instead."
        )

    # Download/cache model BEFORE initializing Ray
    # This ensures model files are local before pipeline actors try to load them
    if use_vllm:
        logger.info(
            "Downloading model for vLLM (this may take a while on first run)..."
        )
        model_path = download_model_for_vllm(model_name, models_dir)
        logger.info(f"Using vLLM backend with local model: {model_path}")
    elif not skip_cache:
        logger.info("Caching model before starting pipeline...")
        model_path = cache_model(model_name, dtype, cache_dir)
        logger.info(f"Using cached model at: {model_path}")
    else:
        model_path = model_name
        logger.info(f"Skipping cache, using model directly: {model_path}")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    logger.info("Ray initialized")

    # Verify GPU allocation works correctly
    @ray.remote(num_gpus=1)
    def check_gpu():
        import os
        import torch

        return {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_count": torch.cuda.device_count(),
        }

    logger.info("Checking GPU allocation...")
    gpu_tasks = [check_gpu.remote() for _ in range(num_detector_actors)]
    gpu_results = ray.get(gpu_tasks)
    for i, r in enumerate(gpu_results):
        logger.info(f"GPU check - Actor {i}: {r}")

    # Create job queue
    job_queue = Queue(maxsize=1000)
    logger.info("Job queue created")

    # Get audio files
    audio_dir = Path(__file__).parent.parent / data_dir
    audio_files = [f for f in audio_dir.glob("*.mp3") if f.is_file()]

    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        ray.shutdown()
        return

    logger.info(f"Found {len(audio_files)} audio files to sample from")

    # Create the streaming datasource from the job queue
    datasource = QueueStreamingDatasource(
        queue=job_queue,
        item_to_row_fn=lambda x: x,  # Items are already dicts from job_to_row()
        config=StreamingDatasourceConfig(
            parallelism=datasource_parallelism,
            batch_size=datasource_batch_size,
            batch_timeout=0.1,
            poll_interval=0.01,
            max_items=total_jobs,  # Stop after processing all jobs
        ),
    )
    logger.info("StreamingDatasource created")

    # Create agent stages
    preprocessor_stage = AgentStage(
        agent=AudioPreprocessorAgent(target_sr=16000),
        config=AgentRayComputeConfig(
            num_actors=num_preprocessor_actors,
            batch_size=preprocessor_batch_size,
            num_cpus=preprocessor_num_cpus,
            max_concurrency=preprocessor_max_concurrency,
        ),
        name="AudioPreprocessor",
    )

    # Choose agent class based on use_cot flag
    # Note: tensor_parallel_size is now passed to both agent and config
    # The config uses it to calculate num_gpus, the agent passes it to vLLM
    if use_cot:
        detector_agent = InstrumentDetectorCoTAgent(
            model_name=model_path,
            dtype=dtype,
            use_vllm=use_vllm,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )
    else:
        detector_agent = InstrumentDetectorAgent(
            model_name=model_path,
            dtype=dtype,
            use_vllm=use_vllm,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )

    # Configure detector stage with tensor parallelism support
    # When tensor_parallel_size > 1, num_gpus is calculated automatically
    detector_config = AgentRayComputeConfig(
        num_actors=num_detector_actors,
        batch_size=detector_batch_size,
        num_gpus=detector_num_gpus,  # Used if tensor_parallel_size=1
        max_concurrency=detector_max_concurrency,
        # Tensor parallelism - num_gpus will be overridden to tensor_parallel_size
        tensor_parallel_size=tensor_parallel_size or 1,
    )

    logger.info(
        f"Detector config: num_actors={num_detector_actors}, "
        f"tensor_parallel_size={tensor_parallel_size or 1}, "
        f"calculated_gpus={detector_config.get_num_gpus_per_actor()}"
    )

    detector_stage = AgentStage(
        agent=detector_agent,
        config=detector_config,
        name="InstrumentDetector",
    )

    # Create the streaming pipeline
    pipeline = StreamingPipeline(
        datasource=datasource,
        stages=[preprocessor_stage, detector_stage],
        name="InstrumentDetectionPipeline",
    )
    logger.info("StreamingPipeline created")

    # Setup shutdown handling
    shutdown_requested = threading.Event()

    def signal_handler(signum, _frame):
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"Received {sig_name}, initiating shutdown...")
        shutdown_requested.set()
        datasource.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create the streaming job producer (don't start yet)
    producer = StreamingJobProducer(
        job_queue=job_queue,
        audio_files=audio_files,
        total_jobs=total_jobs,
        min_delay_ms=min_delay_ms,
        max_delay_ms=max_delay_ms,
        burst_probability=burst_probability,
        burst_size_min=burst_size_min,
        burst_size_max=burst_size_max,
    )

    # Warmup function - creates warmup data from a real audio file
    def get_warmup_data():
        """Create warmup data using a real audio file."""
        warmup_file = audio_files[0]
        audio_bytes = warmup_file.read_bytes()
        return [
            {
                "job_id": "warmup_001",
                "song_id": "warmup",
                "song_hash": "warmup",
                "filename": warmup_file.name,
                "audio_bytes": audio_bytes,
            }
        ]

    # Use warmup_and_stream to ensure the same actors handle both warmup and real data
    # This is CRITICAL for vLLM - model loads once during warmup, then same actors process real data
    logger.info(
        "Starting warmup_and_stream (model will load on first batch, may take a few minutes)..."
    )

    # Get the streaming iterator with warmup
    # Warmup items are injected into the queue FIRST
    streaming_iterator = pipeline.warmup_and_stream(
        warmup_data_fn=get_warmup_data,
        warmup_timeout=300.0,
        batch_size=1,
    )

    # NOW start the producer - its jobs will come AFTER warmup items in the queue
    logger.info("Starting streaming job producer...")
    producer.start()

    # Stream results from the pipeline
    results = []
    start_time = time.time()
    last_log_time = start_time

    logger.info(
        "Streaming results (warmup items will be processed first and discarded)..."
    )

    try:
        for batch in streaming_iterator:
            if shutdown_requested.is_set():
                break

            # Extract results from the batch
            # batch is a dict of columns, convert to list of dicts
            if batch:
                keys = list(batch.keys())
                if keys:
                    n_items = len(batch[keys[0]])
                    for i in range(n_items):
                        result = {k: batch[k][i] for k in keys}
                        results.append(result)

                        # Log result with error handling
                        if result.get("error"):
                            logger.warning(
                                f"Result: {result['filename']} → ERROR: {result['error']}"
                            )
                        elif result.get("background") is not None:
                            # CoT output with layers
                            logger.info(
                                f"Result: {result['filename']} → "
                                f"BG: {result.get('background', [])}, "
                                f"MG: {result.get('middle_ground', [])}, "
                                f"FG: {result.get('foreground', [])}"
                            )
                        else:
                            logger.info(
                                f"Result: {result['filename']} → {result['instruments']}"
                            )

            # Log progress periodically
            if time.time() - last_log_time >= 1.0:
                logger.info(
                    f"Progress: submitted={producer.jobs_submitted}/{total_jobs}, "
                    f"consumed={len(results)}"
                )
                last_log_time = time.time()

            # Check if done
            if len(results) >= total_jobs:
                logger.info("All items processed!")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback

        traceback.print_exc()

    # Stop producer and pipeline
    producer.stop()
    pipeline.stop()

    end_time = time.time()
    total_time = end_time - start_time

    # Final stats
    successful = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    logger.info("=" * 60)
    logger.info("STREAMING PIPELINE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total jobs submitted: {producer.jobs_submitted}")
    logger.info(f"Total results received: {len(results)}")
    logger.info(f"  - Successful: {len(successful)}")
    logger.info(f"  - Failed: {len(failed)}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    if len(successful) > 0:
        logger.info(f"Throughput: {len(successful) / total_time:.2f} files/second")
        logger.info(f"Avg time per file: {total_time / len(successful):.2f} seconds")

    # Show sample results
    logger.info("\nSample results:")
    for result in successful[:5]:
        if result.get("background") is not None:
            logger.info(
                f"  - {result['filename']}: BG={result.get('background', [])}, "
                f"MG={result.get('middle_ground', [])}, FG={result.get('foreground', [])}"
            )
        else:
            logger.info(f"  - {result['filename']}: {result['instruments']}")

    # Show errors if any
    if failed:
        logger.info("\nFailed items:")
        for result in failed[:5]:
            logger.info(f"  - {result['filename']}: {result['error']}")

    # Cleanup
    ray.shutdown()
    logger.info("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test streaming instrument detection pipeline"
    )

    # Job producer config
    parser.add_argument(
        "--data-dir",
        type=str,
        default="audio_files",
        help="Directory to load audio files from",
    )
    parser.add_argument(
        "--total-jobs", type=int, default=50, help="Total number of jobs to submit"
    )
    parser.add_argument(
        "--min-delay", type=int, default=100, help="Min delay between jobs (ms)"
    )
    parser.add_argument(
        "--max-delay", type=int, default=2000, help="Max delay between jobs (ms)"
    )
    parser.add_argument(
        "--burst-prob", type=float, default=0.2, help="Probability of burst submission"
    )
    parser.add_argument("--burst-min", type=int, default=3, help="Min jobs in a burst")
    parser.add_argument("--burst-max", type=int, default=10, help="Max jobs in a burst")

    # Datasource config
    parser.add_argument("--ds-batch", type=int, default=8, help="Datasource batch size")
    parser.add_argument(
        "--ds-parallelism", type=int, default=1, help="Datasource parallelism"
    )

    # Preprocessor config
    parser.add_argument(
        "--num-preprocessors", type=int, default=4, help="Number of preprocessor actors"
    )
    parser.add_argument(
        "--prep-batch", type=int, default=8, help="Preprocessor batch size"
    )
    parser.add_argument(
        "--prep-cpus", type=float, default=1.0, help="CPUs per preprocessor"
    )
    parser.add_argument(
        "--prep-concurrency",
        type=int,
        default=1,
        help="Max concurrency per preprocessor",
    )

    # Detector config
    # IMPORTANT: For vLLM with multi-GPU, use 1 detector with tensor parallelism.
    # DO NOT use multiple detectors with vLLM - each would try to load the full model.
    # Use: --num-detectors 1 --tensor-parallel-size 4 --detector-gpus 4.0
    parser.add_argument(
        "--num-detectors",
        type=int,
        default=1,
        help="Number of detector actors (use 1 for vLLM multi-GPU)",
    )
    parser.add_argument(
        "--detector-batch", type=int, default=4, help="Detector batch size"
    )
    parser.add_argument(
        "--detector-gpus",
        type=float,
        default=1.0,
        help="GPUs per detector (set to num_gpus for tensor parallelism)",
    )
    parser.add_argument(
        "--detector-concurrency",
        type=int,
        default=1,
        help="Max concurrency per detector",
    )

    # Model config
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name",
    )

    # Cache config
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Directory to cache HuggingFace models",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help="Directory to download vLLM models",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip caching and use model name directly",
    )

    # Model dtype
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (float32, float16, bfloat16)",
    )

    # vLLM options
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM backend instead of HuggingFace",
    )
    parser.add_argument(
        "--use-cot",
        action="store_true",
        help="Use chain-of-thought (CoT) agent for layered detection",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism (default: auto)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization for vLLM (0.0-1.0)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model context length for vLLM",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=8,
        help="Maximum number of sequences for vLLM",
    )

    args = parser.parse_args()

    print(
        f"""
Streaming Pipeline Configuration:
  Producer: total_jobs={args.total_jobs}, delay={args.min_delay}-{args.max_delay}ms, burst_prob={args.burst_prob}, burst_size={args.burst_min}-{args.burst_max}
  Datasource: batch_size={args.ds_batch}, parallelism={args.ds_parallelism}
  Preprocessor: num_actors={args.num_preprocessors}, batch={args.prep_batch}, cpus={args.prep_cpus}, concurrency={args.prep_concurrency}
  Detector: num_actors={args.num_detectors}, batch={args.detector_batch}, gpus={args.detector_gpus}, concurrency={args.detector_concurrency}
  Model: {args.model}
  Cache: cache_dir={args.cache_dir}, models_dir={args.models_dir}, skip_cache={args.skip_cache}
  Dtype: {args.dtype}
  Backend: {"vLLM" if args.use_vllm else "HuggingFace"}, CoT: {args.use_cot}
  vLLM: tp_size={args.tensor_parallel_size}, gpu_mem={args.gpu_memory_utilization}, max_model_len={args.max_model_len}, max_num_seqs={args.max_num_seqs}
"""
    )

    main(
        data_dir=args.data_dir,
        total_jobs=args.total_jobs,
        min_delay_ms=args.min_delay,
        max_delay_ms=args.max_delay,
        burst_probability=args.burst_prob,
        burst_size_min=args.burst_min,
        burst_size_max=args.burst_max,
        datasource_batch_size=args.ds_batch,
        datasource_parallelism=args.ds_parallelism,
        num_preprocessor_actors=args.num_preprocessors,
        preprocessor_batch_size=args.prep_batch,
        preprocessor_num_cpus=args.prep_cpus,
        preprocessor_max_concurrency=args.prep_concurrency,
        num_detector_actors=args.num_detectors,
        detector_batch_size=args.detector_batch,
        detector_num_gpus=args.detector_gpus,
        detector_max_concurrency=args.detector_concurrency,
        model_name=args.model,
        cache_dir=args.cache_dir,
        models_dir=args.models_dir,
        skip_cache=args.skip_cache,
        dtype=DTYPE_MAP[args.dtype],
        # vLLM options
        use_vllm=args.use_vllm,
        use_cot=args.use_cot,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
