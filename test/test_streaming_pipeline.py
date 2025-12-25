"""
Test script for streaming instrument detection pipeline.
Jobs are added to the queue at random intervals while the pipeline is consuming.
This simulates a real-time production environment.
"""

import ray
import time
import signal
import random
from pathlib import Path
from loguru import logger
from ray.util.queue import Queue, Empty

from src.instrument_detect.job_queue import create_job_queue
from src.instrument_detect.pipeline import InstrumentDetectPipeline
from src.instrument_detect.data_classes import InstrumentDetectJob
import hashlib
import uuid
import threading

# Default cache directory
DEFAULT_CACHE_DIR = "/app/cache"


def cache_model(model_name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
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

    logger.info("Loading model from HuggingFace...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)

    logger.info(f"Saving model to {cache_path}")
    model.save_pretrained(cache_path)
    processor.save_pretrained(cache_path)

    logger.info(f"Model cached successfully at {cache_path}")
    return str(cache_path)


def create_job_from_file(filepath: Path) -> InstrumentDetectJob:
    """Create a job from a file path."""
    audio_bytes = filepath.read_bytes()
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    song_id = f"trk_{uuid.uuid4().hex[:12]}"
    song_hash = hashlib.sha256(audio_bytes).hexdigest()
    created_at = int(time.time())

    audio_ref = ray.put(audio_bytes)

    return InstrumentDetectJob(
        job_id=job_id,
        created_at=created_at,
        song_id=song_id,
        song_hash=song_hash,
        audio_ref=audio_ref,
        filename=filepath.name,
    )


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
        self.job_queue.put(job)
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

            if self.jobs_submitted % 10 == 0:
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
    total_jobs: int = 50,
    min_delay_ms: int = 100,
    max_delay_ms: int = 2000,
    burst_probability: float = 0.2,
    burst_size_min: int = 3,
    burst_size_max: int = 10,
    # Preprocessor config
    pool_size: int = 4,
    dispatcher_batch_size: int = 8,
    preprocessor_num_cpus: float = 1.0,
    preprocessor_max_concurrency: int = 1,
    # Detector config
    num_detector_actors: int = 1,
    detector_batch_size: int = 4,
    detector_num_gpus: float = 1.0,
    detector_max_concurrency: int = 1,
    # Pipeline config
    max_pending_tasks: int = 16,
    max_waveform_queue_size: int = 50,
    result_queue_size: int = 100,
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    # Cache config
    cache_dir: str = DEFAULT_CACHE_DIR,
    skip_cache: bool = False,
):
    # Cache model before initializing Ray
    if not skip_cache:
        logger.info("Caching model before starting pipeline...")
        model_path = cache_model(model_name, cache_dir)
        logger.info(f"Using cached model at: {model_path}")
    else:
        model_path = model_name
        logger.info(f"Skipping cache, using model directly: {model_path}")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    logger.info("Ray initialized")

    # Create job queue
    job_queue = create_job_queue(max_size=1000)
    logger.info("Job queue created")

    # Get audio files
    audio_dir = Path(__file__).parent.parent / "audio_files"
    audio_files = [f for f in audio_dir.glob("*.mp3") if f.is_file()]

    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        ray.shutdown()
        return

    logger.info(f"Found {len(audio_files)} audio files to sample from")

    # Create pipeline
    logger.info("Creating pipeline...")
    pipeline = InstrumentDetectPipeline(
        job_queue=job_queue,
        pool_size=pool_size,
        dispatcher_batch_size=dispatcher_batch_size,
        detector_batch_size=detector_batch_size,
        max_pending_tasks=max_pending_tasks,
        max_waveform_queue_size=max_waveform_queue_size,
        result_queue_size=result_queue_size,
        model_name=model_path,
        preprocessor_num_cpus=preprocessor_num_cpus,
        preprocessor_max_concurrency=preprocessor_max_concurrency,
        detector_num_gpus=detector_num_gpus,
        detector_max_concurrency=detector_max_concurrency,
        num_detector_actors=num_detector_actors,
    )

    # Setup shutdown handling
    shutdown_requested = threading.Event()

    def signal_handler(signum, _frame):
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"Received {sig_name}, initiating shutdown...")
        shutdown_requested.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start pipeline FIRST (before adding any jobs)
    logger.info("Starting pipeline...")
    pipeline.start()

    # Start result consumer thread
    results = []
    consumer_done = threading.Event()

    def consume_results():
        while not consumer_done.is_set():
            try:
                result = pipeline.result_queue.get(timeout=0.5)
                results.append(result)
                logger.info(f"Result: {result.filename} → {result.instruments}")
            except Empty:
                continue

    consumer_thread = threading.Thread(target=consume_results, daemon=True)
    consumer_thread.start()
    logger.info("Result consumer started")

    # Create and start the streaming job producer
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

    logger.info("Starting streaming job producer...")
    producer.start()

    # Monitor progress
    start_time = time.time()
    last_log_time = start_time

    try:
        while not shutdown_requested.is_set():
            # Check for failures
            error = pipeline.check_for_failures(timeout=0)
            if error:
                logger.error(f"Pipeline failure: {error}")
                break

            # Get stats
            try:
                stats = pipeline.get_stats()
                job_size = stats["job_queue_size"]
                waveform_size = stats["waveform_queue_size"]
                result_size = stats["result_queue_size"]
                dispatcher_stats = stats.get("dispatcher", {})
                detector_stats = stats.get("detectors", [{}])
            except Exception:
                job_size = waveform_size = result_size = "?"
                dispatcher_stats = {}
                detector_stats = [{}]

            # Log every second
            if time.time() - last_log_time >= 1.0:
                dispatched = dispatcher_stats.get("dispatched_count", "?")
                pending = dispatcher_stats.get("pending_tasks", "?")
                detected = sum(
                    d.get("processed_count", 0)
                    for d in detector_stats
                    if isinstance(d, dict)
                )

                logger.info(
                    f"Progress: submitted={producer.jobs_submitted}/{total_jobs}, "
                    f"jobs_queue={job_size}, waveforms={waveform_size}, results={result_size}, "
                    f"dispatched={dispatched}, pending={pending}, detected={detected}, consumed={len(results)}"
                )
                last_log_time = time.time()

            # Check if done (all jobs submitted AND all results received)
            if producer.is_done() and len(results) >= total_jobs:
                logger.info("All items processed!")
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Interrupted")

    # Stop producer
    producer.stop()

    # Stop consumer
    consumer_done.set()
    consumer_thread.join(timeout=2.0)

    end_time = time.time()
    total_time = end_time - start_time

    # Final stats
    logger.info("=" * 60)
    logger.info("STREAMING PIPELINE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total jobs submitted: {producer.jobs_submitted}")
    logger.info(f"Total results received: {len(results)}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    if len(results) > 0:
        logger.info(f"Throughput: {len(results) / total_time:.2f} files/second")
        logger.info(f"Avg time per file: {total_time / len(results):.2f} seconds")

    # Detector timing metrics
    try:
        stats = pipeline.get_stats()
        detector_stats = stats.get("detectors", [])
        if detector_stats:
            logger.info("\nDetector Inference Timing:")
            for i, ds in enumerate(detector_stats):
                if isinstance(ds, dict):
                    logger.info(
                        f"  Detector {i}: {ds.get('batch_count', 0)} batches, "
                        f"{ds.get('processed_count', 0)} examples, "
                        f"avg {ds.get('avg_batch_time_ms', 0):.1f}ms/batch, "
                        f"{ds.get('avg_per_example_ms', 0):.1f}ms/example"
                    )
    except Exception as e:
        logger.warning(f"Could not get detector timing stats: {e}")

    # Show sample results
    logger.info("\nSample results:")
    for result in results[:5]:
        logger.info(f"  - {result.filename}: {result.instruments}")

    # Shutdown pipeline
    logger.info("\nShutting down pipeline...")
    pipeline.shutdown()

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

    # Preprocessor config
    parser.add_argument(
        "--pool-size", type=int, default=4, help="Number of preprocessor actors"
    )
    parser.add_argument(
        "--dispatcher-batch", type=int, default=8, help="Dispatcher batch size"
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
    parser.add_argument(
        "--num-detectors", type=int, default=1, help="Number of detector actors"
    )
    parser.add_argument(
        "--detector-batch", type=int, default=4, help="Detector batch size"
    )
    parser.add_argument(
        "--detector-gpus", type=float, default=1.0, help="GPUs per detector"
    )
    parser.add_argument(
        "--detector-concurrency",
        type=int,
        default=1,
        help="Max concurrency per detector",
    )

    # Pipeline config
    parser.add_argument(
        "--max-pending", type=int, default=16, help="Max pending preprocessor tasks"
    )
    parser.add_argument(
        "--max-waveform-queue", type=int, default=50, help="Max waveform queue size"
    )
    parser.add_argument(
        "--result-queue-size", type=int, default=100, help="Result queue size"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
        help="Model name",
    )

    # Cache config
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Directory to cache models",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip caching and use model name directly",
    )

    args = parser.parse_args()

    print(
        f"""
Streaming Pipeline Configuration:
  Producer: total_jobs={args.total_jobs}, delay={args.min_delay}-{args.max_delay}ms, burst_prob={args.burst_prob}, burst_size={args.burst_min}-{args.burst_max}
  Preprocessor: pool_size={args.pool_size}, batch={args.dispatcher_batch}, cpus={args.prep_cpus}, concurrency={args.prep_concurrency}
  Detector: num_actors={args.num_detectors}, batch={args.detector_batch}, gpus={args.detector_gpus}, concurrency={args.detector_concurrency}
  Pipeline: max_pending={args.max_pending}, max_waveform_queue={args.max_waveform_queue}, model={args.model}
  Cache: dir={args.cache_dir}, skip_cache={args.skip_cache}
"""
    )

    main(
        total_jobs=args.total_jobs,
        min_delay_ms=args.min_delay,
        max_delay_ms=args.max_delay,
        burst_probability=args.burst_prob,
        burst_size_min=args.burst_min,
        burst_size_max=args.burst_max,
        pool_size=args.pool_size,
        dispatcher_batch_size=args.dispatcher_batch,
        preprocessor_num_cpus=args.prep_cpus,
        preprocessor_max_concurrency=args.prep_concurrency,
        num_detector_actors=args.num_detectors,
        detector_batch_size=args.detector_batch,
        detector_num_gpus=args.detector_gpus,
        detector_max_concurrency=args.detector_concurrency,
        max_pending_tasks=args.max_pending,
        max_waveform_queue_size=args.max_waveform_queue,
        result_queue_size=args.result_queue_size,
        model_name=args.model,
        cache_dir=args.cache_dir,
        skip_cache=args.skip_cache,
    )
