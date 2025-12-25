"""
Test script for the full instrument detection pipeline.
Tests preprocessor → detector flow end-to-end.
"""

import ray
import time
import signal
import sys
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

    # Create cache path from model name (replace / with _)
    model_cache_name = model_name.replace("/", "_")
    cache_path = Path(cache_dir) / model_cache_name

    if cache_path.exists():
        logger.info(f"Model already cached at {cache_path}")
        return str(cache_path)

    logger.info(f"Downloading and caching model {model_name} to {cache_path}")
    cache_path.mkdir(parents=True, exist_ok=True)

    # Load model and processor from HuggingFace
    logger.info("Loading model from HuggingFace...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)

    # Save to cache
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


def populate_job_queue(
    job_queue: Queue, audio_dir: Path, multiplier: int = 1, num_workers: int = 4
):
    """Populate job queue with audio files in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    audio_files = [f for f in audio_dir.glob("*.mp3") if f.is_file()]

    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        return 0

    # Create list of all files to process (with multiplier)
    all_files = audio_files * multiplier
    total_files = len(all_files)
    logger.info(
        f"Adding {total_files} items to queue (multiplier={multiplier}x, workers={num_workers})"
    )

    start_time = time.time()
    added_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(create_job_from_file, f): f for f in all_files}

        for future in as_completed(futures):
            try:
                job = future.result()
                job_queue.put(job)
                added_count += 1

                if added_count % 100 == 0:
                    logger.info(f"Added {added_count}/{total_files} jobs")
            except Exception as e:
                logger.error(f"Error creating job: {e}")

    end_time = time.time()
    logger.info(f"Time taken to populate queue: {end_time - start_time:.2f} seconds")

    return total_files


def main(
    # Job queue config
    queue_multiplier: int = 1,
    job_queue_workers: int = 4,
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
    # Cache model before initializing Ray (so it's available to all actors)
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
    job_queue = create_job_queue(max_size=queue_multiplier * 1000)
    logger.info("Job queue created")

    # Get audio files directory
    audio_dir = Path(__file__).parent.parent / "audio_files"

    # Populate job queue
    total_files = populate_job_queue(
        job_queue, audio_dir, queue_multiplier, num_workers=job_queue_workers
    )
    if total_files == 0:
        ray.shutdown()
        return

    logger.info(f"Job queue size: {job_queue.qsize()}")

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

    # Start pipeline
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
                    f"Progress: jobs={job_size}, waveforms={waveform_size}, results={result_size}, "
                    f"dispatched={dispatched}, pending={pending}, detected={detected}, consumed={len(results)}"
                )
                last_log_time = time.time()

            # Check if done
            if len(results) >= total_files:
                logger.info("All items processed!")
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Interrupted")

    # Stop consumer
    consumer_done.set()
    consumer_thread.join(timeout=2.0)

    end_time = time.time()
    total_time = end_time - start_time

    # Final stats
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total files submitted: {total_files}")
    logger.info(f"Total results received: {len(results)}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    if len(results) > 0:
        logger.info(f"Throughput: {len(results) / total_time:.2f} files/second")
        logger.info(f"Time per file: {total_time / len(results):.2f} seconds")

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
        description="Test full instrument detection pipeline"
    )

    # Job queue config
    parser.add_argument(
        "--multiplier", type=int, default=1, help="File multiplier for queue"
    )
    parser.add_argument(
        "--job-workers", type=int, default=4, help="Workers to populate job queue"
    )

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
Pipeline Configuration:
  Job Queue: multiplier={args.multiplier}, workers={args.job_workers}
  Preprocessor: pool_size={args.pool_size}, batch={args.dispatcher_batch}, cpus={args.prep_cpus}, concurrency={args.prep_concurrency}
  Detector: num_actors={args.num_detectors}, batch={args.detector_batch}, gpus={args.detector_gpus}, concurrency={args.detector_concurrency}
  Pipeline: max_pending={args.max_pending}, max_waveform_queue={args.max_waveform_queue}, model={args.model}
  Cache: dir={args.cache_dir}, skip_cache={args.skip_cache}
"""
    )

    main(
        queue_multiplier=args.multiplier,
        job_queue_workers=args.job_workers,
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
