"""
Test script for the preprocessor pipeline without the detector.
Populates job queue and verifies waveform queue gets populated.
"""

import ray
import time
from pathlib import Path
from loguru import logger
from ray.util.queue import Queue, Empty

from src.instrument_detect.job_queue import create_job_queue
from src.instrument_detect.pipeline import PreprocessorDispatcher
from src.instrument_detect.data_classes import InstrumentDetectJob, PreprocessedAudio
import hashlib
import uuid


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
    job_queue: Queue, audio_dir: Path, multiplier: int = 1, num_workers: int = 16
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


def drain_waveform_queue(
    waveform_queue: Queue, expected_count: int, timeout: float = 60.0
):
    """Drain waveform queue and verify items."""
    results = []
    start_time = time.time()

    while len(results) < expected_count:
        if time.time() - start_time > timeout:
            logger.warning(f"Timeout! Got {len(results)}/{expected_count} items")
            break

        try:
            item = waveform_queue.get(timeout=1.0)
            results.append(item)
            logger.info(
                f"Got preprocessed: {item.filename} (waveform shape: {item.waveform.shape})"
            )
        except Empty:
            continue

    return results


def main(
    queue_multiplier: int = 1,
    pool_size: int = 4,
    job_queue_workers: int = 4,
    batch_size: int = 8,
    num_cpus: float = 1.0,
    max_concurrency: int = 1,
):
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    logger.info("Ray initialized")

    # Create queues
    job_queue = create_job_queue(max_size=queue_multiplier * 1000)
    waveform_queue = Queue(maxsize=100)
    logger.info("Queues created")

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

    # Create and start dispatcher (preprocessor only, no detector)
    logger.info(
        f"Starting Dispatcher with pool_size={pool_size}, "
        f"num_cpus={num_cpus}, max_concurrency={max_concurrency}"
    )
    dispatcher = PreprocessorDispatcher.remote(
        job_queue=job_queue,
        waveform_queue=waveform_queue,
        pool_size=pool_size,
        batch_size=batch_size,
        max_pending_tasks=1000,
        max_waveform_queue_size=1000,
        preprocessor_num_cpus=num_cpus,
        preprocessor_max_concurrency=max_concurrency,
    )
    dispatcher.run_forever.remote()
    logger.info("Dispatcher started")

    # Start consumer thread to drain waveform queue (simulating detector)
    import threading

    consumed_items = []
    consumer_done = threading.Event()

    def consume_waveforms():
        while not consumer_done.is_set():
            try:
                item = waveform_queue.get(timeout=0.1)
                consumed_items.append(item)
            except Empty:
                continue

    consumer_thread = threading.Thread(target=consume_waveforms, daemon=True)
    consumer_thread.start()
    logger.info("Consumer thread started")

    # Monitor progress
    start_time = time.time()
    last_log_time = start_time

    while True:
        job_size = job_queue.qsize()
        waveform_size = waveform_queue.qsize()

        try:
            stats = ray.get(dispatcher.get_stats.remote(), timeout=1.0)
            dispatched = stats["dispatched_count"]
            pending = stats["pending_tasks"]
        except:
            dispatched = "?"
            pending = "?"

        # Log every second
        if time.time() - last_log_time >= 1.0:
            logger.info(
                f"Progress: job_queue={job_size}, waveform_queue={waveform_size}, "
                f"dispatched={dispatched}, pending={pending}, consumed={len(consumed_items)}"
            )
            last_log_time = time.time()

        # Check if done - all items consumed
        if isinstance(dispatched, int) and len(consumed_items) >= total_files:
            break

        time.sleep(0.1)

    # Stop consumer
    consumer_done.set()
    consumer_thread.join(timeout=2.0)

    end_time = time.time()
    total_time = end_time - start_time

    # Final stats
    logger.info("=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total files processed: {total_files}")
    logger.info(f"Total consumed: {len(consumed_items)}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Throughput: {total_files / total_time:.2f} files/second")
    logger.info(f"Time per file: {total_time / total_files:.2f} seconds")
    logger.info(f"Final waveform queue size: {waveform_queue.qsize()}")

    # Verify some consumed items
    logger.info("\nSample consumed items:")
    for item in consumed_items[:3]:
        logger.info(f"  - {item.filename}: waveform shape {item.waveform.shape}")

    # Cleanup
    ray.shutdown()
    logger.info("Done!")


if __name__ == "__main__":
    import sys

    multiplier = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    pool_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    job_queue_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    num_cpus = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    max_concurrency = int(sys.argv[6]) if len(sys.argv) > 6 else 1

    print(
        f"""
Usage: python {sys.argv[0]} [multiplier] [pool_size] [job_queue_workers] [batch_size] [num_cpus] [max_concurrency]
  multiplier={multiplier}, pool_size={pool_size}, job_queue_workers={job_queue_workers}
  batch_size={batch_size}, num_cpus={num_cpus}, max_concurrency={max_concurrency}
"""
    )

    main(
        queue_multiplier=multiplier,
        pool_size=pool_size,
        job_queue_workers=job_queue_workers,
        batch_size=batch_size,
        num_cpus=num_cpus,
        max_concurrency=max_concurrency,
    )
