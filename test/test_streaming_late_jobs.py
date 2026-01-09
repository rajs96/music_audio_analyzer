"""
Test streaming pipeline with late job additions.

This test verifies that the pipeline can handle jobs added WHILE
the GPU is actively processing earlier jobs.

Flow:
1. Start pipeline with warmup
2. Add first batch of jobs (e.g., 10 jobs)
3. Wait 2 minutes while GPU processes
4. Add second batch of jobs (e.g., 10 more jobs)
5. Wait 2 more minutes
6. Add final batch and stop signal
7. Verify all jobs complete

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
    InstrumentDetectorCoTAgent,
)

# Map string names to torch dtypes
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

DEFAULT_MODELS_DIR = "/app/models"


def download_model_for_vllm(
    model_name: str, models_dir: str = DEFAULT_MODELS_DIR
) -> str:
    """Download model files for vLLM."""
    from huggingface_hub import snapshot_download

    model_cache_name = model_name.replace("/", "_")
    local_path = Path(models_dir) / model_cache_name

    if local_path.exists() and any(local_path.iterdir()):
        logger.info(f"Model already downloaded at {local_path}")
        return str(local_path)

    logger.info(f"Downloading model {model_name} to {local_path}...")
    local_path.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
    )

    logger.info(f"Model downloaded successfully to {local_path}")
    return str(local_path)


def create_job_from_file(filepath: Path) -> dict:
    """Create a job dict from a file path."""
    audio_bytes = filepath.read_bytes()
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    song_id = f"trk_{uuid.uuid4().hex[:12]}"
    song_hash = hashlib.sha256(audio_bytes).hexdigest()

    return {
        "job_id": job_id,
        "song_id": song_id,
        "song_hash": song_hash,
        "filename": filepath.name,
        "audio_bytes": audio_bytes,
    }


class ResultConsumer(threading.Thread):
    """Thread that consumes results from the streaming iterator."""

    def __init__(self, streaming_iterator, total_expected: int):
        super().__init__(daemon=True)
        self.streaming_iterator = streaming_iterator
        self.total_expected = total_expected
        self.results = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def run(self):
        """Consume results from the iterator."""
        logger.info("ResultConsumer started")

        try:
            for batch in self.streaming_iterator:
                if self.stop_event.is_set():
                    break

                if batch:
                    keys = list(batch.keys())
                    if keys:
                        n_items = len(batch[keys[0]])
                        for i in range(n_items):
                            result = {k: batch[k][i] for k in keys}

                            with self.lock:
                                self.results.append(result)
                                count = len(self.results)

                            if result.get("error") and str(result.get("error")).strip():
                                logger.warning(
                                    f"Result {count}: {result['filename']} -> ERROR: {result['error']}"
                                )
                            else:
                                logger.info(
                                    f"Result {count}: {result['filename']} -> "
                                    f"BG: {result.get('background', [])}, "
                                    f"MG: {result.get('middle_ground', [])}, "
                                    f"FG: {result.get('foreground', [])}"
                                )

                # Check if we have all results
                with self.lock:
                    if len(self.results) >= self.total_expected:
                        logger.info("All expected results received!")
                        break

        except Exception as e:
            logger.error(f"ResultConsumer error: {e}")
            import traceback

            traceback.print_exc()

        logger.info(f"ResultConsumer finished with {len(self.results)} results")

    def stop(self):
        self.stop_event.set()

    def get_results(self):
        with self.lock:
            return list(self.results)

    def get_count(self):
        with self.lock:
            return len(self.results)


def main(
    # Test config
    data_dir: str = "audio_files",
    batch1_size: int = 10,
    batch2_size: int = 10,
    batch3_size: int = 10,
    delay_between_batches_sec: int = 120,  # 2 minutes
    # Model config
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    models_dir: str = DEFAULT_MODELS_DIR,
    dtype: torch.dtype = torch.bfloat16,
    # vLLM config
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 32768,
    max_num_seqs: int = 8,
    # Pipeline config
    detector_batch_size: int = 4,
):
    total_jobs = batch1_size + batch2_size + batch3_size

    # Download model
    logger.info("Downloading model for vLLM...")
    model_path = download_model_for_vllm(model_name, models_dir)
    logger.info(f"Using model at: {model_path}")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    logger.info("Ray initialized")

    # Get audio files
    audio_dir = Path(__file__).parent.parent / data_dir
    audio_files = [f for f in audio_dir.glob("*.mp3") if f.is_file()]

    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        ray.shutdown()
        return 1

    logger.info(f"Found {len(audio_files)} audio files to sample from")

    # Create job queue
    job_queue = Queue(maxsize=1000)
    logger.info("Job queue created")

    # Create datasource
    datasource = QueueStreamingDatasource(
        queue=job_queue,
        item_to_row_fn=lambda x: x,
        config=StreamingDatasourceConfig(
            parallelism=1,
            batch_size=8,
            batch_timeout=0.1,
            poll_interval=0.01,
            max_items=total_jobs,
        ),
    )

    # Create pipeline stages
    preprocessor_stage = AgentStage(
        agent=AudioPreprocessorAgent(target_sr=16000),
        config=AgentRayComputeConfig(
            num_actors=4,
            batch_size=8,
            num_cpus=1.0,
        ),
        name="AudioPreprocessor",
    )

    detector_stage = AgentStage(
        agent=InstrumentDetectorCoTAgent(
            model_name=model_path,
            dtype=dtype,
            use_vllm=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        ),
        config=AgentRayComputeConfig(
            num_actors=1,
            batch_size=detector_batch_size,
            num_gpus=float(tensor_parallel_size),
            tensor_parallel_size=tensor_parallel_size,
        ),
        name="InstrumentDetector",
    )

    # Create pipeline
    pipeline = StreamingPipeline(
        datasource=datasource,
        stages=[preprocessor_stage, detector_stage],
        name="LateJobTestPipeline",
    )
    logger.info("Pipeline created")

    # Shutdown handling
    shutdown_requested = threading.Event()

    def signal_handler(signum, _frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_requested.set()
        datasource.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Warmup data
    def get_warmup_data():
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

    # Start streaming with warmup
    logger.info("Starting pipeline with warmup...")
    streaming_iterator = pipeline.warmup_and_stream(
        warmup_data_fn=get_warmup_data,
        warmup_timeout=300.0,
        batch_size=1,
    )

    # Start result consumer thread
    consumer = ResultConsumer(streaming_iterator, total_jobs)
    consumer.start()

    # Job submission tracking
    jobs_submitted = 0
    start_time = time.time()

    def submit_batch(batch_name: str, num_jobs: int):
        nonlocal jobs_submitted
        logger.info(f"\n{'='*60}")
        logger.info(f"SUBMITTING {batch_name}: {num_jobs} jobs")
        logger.info(f"Current results: {consumer.get_count()}")
        logger.info(f"{'='*60}")

        for _ in range(num_jobs):
            audio_file = random.choice(audio_files)
            job = create_job_from_file(audio_file)
            job_queue.put(job)
            jobs_submitted += 1
            logger.info(f"  Submitted job {jobs_submitted}: {job['filename']}")

        logger.info(f"{batch_name} submitted. Total jobs: {jobs_submitted}")

    try:
        # ============================================
        # BATCH 1: Initial jobs
        # ============================================
        submit_batch("BATCH 1", batch1_size)

        # Wait while GPU processes (consumer thread handles results)
        logger.info(
            f"\nWaiting {delay_between_batches_sec}s while GPU processes batch 1..."
        )
        for i in range(delay_between_batches_sec):
            if shutdown_requested.is_set():
                break
            time.sleep(1)
            if i > 0 and i % 30 == 0:
                logger.info(f"  ...{i}s elapsed, {consumer.get_count()} results so far")

        # ============================================
        # BATCH 2: Late jobs while GPU is processing
        # ============================================
        submit_batch("BATCH 2 (LATE)", batch2_size)

        # Wait while GPU processes
        logger.info(
            f"\nWaiting {delay_between_batches_sec}s while GPU processes batch 2..."
        )
        for i in range(delay_between_batches_sec):
            if shutdown_requested.is_set():
                break
            time.sleep(1)
            if i > 0 and i % 30 == 0:
                logger.info(f"  ...{i}s elapsed, {consumer.get_count()} results so far")

        # ============================================
        # BATCH 3: Final batch
        # ============================================
        submit_batch("BATCH 3 (FINAL)", batch3_size)

        # Wait for all results
        logger.info("\nWaiting for all results to complete...")
        timeout = 600  # 10 minute timeout
        start_wait = time.time()

        while consumer.get_count() < total_jobs:
            if shutdown_requested.is_set():
                break
            if time.time() - start_wait > timeout:
                logger.error("Timeout waiting for results!")
                break

            elapsed = int(time.time() - start_wait)
            if elapsed > 0 and elapsed % 30 == 0:
                logger.info(
                    f"  ...waiting, {consumer.get_count()}/{total_jobs} results "
                    f"({elapsed}s elapsed)"
                )
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()

    # Stop consumer and pipeline
    consumer.stop()
    pipeline.stop()

    end_time = time.time()
    total_time = end_time - start_time

    # Get final results
    results = consumer.get_results()

    # Final stats
    successful = [
        r for r in results if not r.get("error") or str(r.get("error")).strip() == ""
    ]
    failed = [
        r for r in results if r.get("error") and str(r.get("error")).strip() != ""
    ]

    logger.info("\n" + "=" * 60)
    logger.info("LATE JOB TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total jobs submitted: {jobs_submitted}")
    logger.info(f"Total results received: {len(results)}")
    logger.info(f"  - Successful: {len(successful)}")
    logger.info(f"  - Failed: {len(failed)}")
    logger.info(f"Total time: {total_time:.2f} seconds")

    if len(successful) > 0:
        logger.info(f"Throughput: {len(successful) / total_time:.2f} files/second")

    # Show sample results
    if successful:
        logger.info("\nSample successful results:")
        for result in successful[:5]:
            logger.info(
                f"  - {result['filename']}: "
                f"BG={result.get('background', [])}, "
                f"MG={result.get('middle_ground', [])}, "
                f"FG={result.get('foreground', [])}"
            )

    if failed:
        logger.info("\nFailed items:")
        for result in failed[:5]:
            logger.info(f"  - {result['filename']}: {result['error']}")

    ray.shutdown()
    logger.info("Done!")

    # Return exit code
    if len(results) >= total_jobs:
        logger.info("\nTEST PASSED: All jobs processed!")
        return 0
    else:
        logger.error(f"\nTEST FAILED: Only {len(results)}/{total_jobs} jobs processed")
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test streaming pipeline with late job additions"
    )

    # Batch sizes
    parser.add_argument("--batch1", type=int, default=10, help="First batch size")
    parser.add_argument(
        "--batch2", type=int, default=10, help="Second batch size (late)"
    )
    parser.add_argument(
        "--batch3", type=int, default=10, help="Third batch size (final)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=120,
        help="Delay between batches in seconds (default: 120)",
    )

    # Model config
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help="Directory for models",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )

    # vLLM config
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Max model context length",
    )
    parser.add_argument(
        "--detector-batch",
        type=int,
        default=4,
        help="Detector batch size",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="audio_files",
        help="Directory with audio files",
    )

    args = parser.parse_args()

    print(
        f"""
Late Job Streaming Test Configuration:
  Batch 1: {args.batch1} jobs
  Batch 2: {args.batch2} jobs (added after {args.delay}s)
  Batch 3: {args.batch3} jobs (added after another {args.delay}s)
  Total: {args.batch1 + args.batch2 + args.batch3} jobs

  Model: {args.model}
  Tensor Parallel: {args.tensor_parallel_size}
  GPU Memory: {args.gpu_memory_utilization}
"""
    )

    exit_code = main(
        data_dir=args.data_dir,
        batch1_size=args.batch1,
        batch2_size=args.batch2,
        batch3_size=args.batch3,
        delay_between_batches_sec=args.delay,
        model_name=args.model,
        models_dir=args.models_dir,
        dtype=DTYPE_MAP[args.dtype],
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        detector_batch_size=args.detector_batch,
    )

    import sys

    sys.exit(exit_code)
