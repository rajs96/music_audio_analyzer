import sys
import ray
import time
from pathlib import Path
from loguru import logger
from ray import serve

from src.instrument_detect.job_queue import create_job_queue
from src.instrument_detect.producer import InstrumentDetectJobProducer
from src.instrument_detect.instrument_detector.consumer import InstrumentDetectConsumer


def main(queue_multiplier: int = 1):
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    serve.start(detached=False)
    logger.info("Ray initialized")

    # Create queue (larger max_size if large mode)
    max_size = queue_multiplier * 1000
    queue = create_job_queue(max_size=max_size)
    logger.info(f"Queue created with max_size={max_size}")

    # Create and deploy producer
    producer = InstrumentDetectJobProducer.bind(queue)
    handle = serve.run(producer)
    logger.info("Producer deployed")

    # Get audio files
    audio_dir = Path(__file__).parent.parent / "audio_files"
    audio_files = [f for f in audio_dir.glob("*.mp3") if f.is_file()]

    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        serve.shutdown()
        ray.shutdown()
        return

    logger.info(f"Found {len(audio_files)} audio files")

    total_files = len(audio_files) * queue_multiplier
    logger.info(f"Adding {total_files} items to queue (multiplier={queue_multiplier}x)")

    start_time = time.time()
    # Add files to queue using add_from_filename
    for i in range(queue_multiplier):
        for filepath in audio_files:
            result_ref = handle.add_from_filename.remote(str(filepath))
            result = (
                result_ref.result()
            )  # gets the actual object result, not a reference
            logger.info(f"Completed round {i + 1}/{queue_multiplier}")

    end_time = time.time()
    logger.info(f"Time taken to populate queue: {end_time - start_time} seconds")

    logger.info(f"Queue size after adding: {queue.qsize()}")

    start_time = time.time()
    # Create consumer with batch_size=4
    batch_size = 8
    consumer = InstrumentDetectConsumer.remote(
        queue=queue,
        worker=None,  # No actual model for testing
        batch_size=8,
        max_wait_ms=25,
    )
    logger.info(f"Consumer created with batch_size={batch_size}")

    # Start consumer in background
    consumer.run_forever.remote()
    logger.info("Consumer started")

    # Monitor queue until empty
    while queue.qsize() > 0:
        logger.info(f"Queue size: {queue.qsize()}")
        time.sleep(0.5)

    logger.info("Queue is empty - all items consumed")

    end_time = time.time()
    logger.info(f"Time taken to consume queue: {end_time - start_time} seconds")
    logger.info(f"Time taken per item: {(end_time - start_time) / total_files} seconds")

    # Cleanup
    serve.shutdown()
    ray.shutdown()
    logger.info("Done!")


if __name__ == "__main__":
    main(queue_multiplier=20)
