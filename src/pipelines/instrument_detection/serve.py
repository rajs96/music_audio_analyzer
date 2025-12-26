"""
Production entrypoint for the Instrument Detection Pipeline.

This script starts both:
1. The FileUploader Ray Serve deployment (HTTP API for file uploads)
2. The StreamingPipeline consumer (processes jobs from the queue)

Usage:
    python -m src.pipelines.instrument_detection.serve

    # Or with custom config:
    python -m src.pipelines.instrument_detection.serve \
        --num-preprocessors 4 \
        --num-detectors 2 \
        --detector-gpus 1.0

API Endpoints (default port 8000):
    POST /upload          - Upload single audio file
    POST /upload/batch    - Upload multiple audio files
    GET  /health          - Health check and queue status
"""

import argparse
import signal
import threading
import time

import ray
from loguru import logger
from ray import serve
from ray.util.queue import Queue

from .file_uploader import FileUploader
from .pipeline import create_pipeline, result_from_row


def run_pipeline_consumer(
    pipeline,
    results_callback=None,
    shutdown_event=None,
):
    """
    Run the pipeline consumer loop.

    Args:
        pipeline: StreamingPipeline instance
        results_callback: Optional callback for each result
        shutdown_event: Threading event to signal shutdown
    """
    logger.info("Starting pipeline consumer...")

    successful = 0
    failed = 0

    try:
        for batch in pipeline.stream(batch_size=1):
            if shutdown_event and shutdown_event.is_set():
                logger.info("Shutdown requested, stopping consumer")
                break

            # Process batch
            if batch:
                keys = list(batch.keys())
                if keys:
                    n_items = len(batch[keys[0]])
                    for i in range(n_items):
                        row = {k: batch[k][i] for k in keys}

                        if row.get("error"):
                            failed += 1
                            logger.warning(
                                f"Failed: {row['filename']} - {row['error']}"
                            )
                        else:
                            successful += 1
                            result = result_from_row(row)
                            logger.info(
                                f"Detected: {result.filename} -> {result.instruments}"
                            )

                            if results_callback:
                                results_callback(result)

    except Exception as e:
        logger.error(f"Pipeline consumer error: {e}")
        raise
    finally:
        logger.info(
            f"Pipeline consumer stopped. Processed: {successful} successful, {failed} failed"
        )


def main(
    # Queue config
    queue_max_size: int = 1000,
    # FileUploader config
    num_uploader_replicas: int = 2,
    max_file_size_mb: float = 100.0,
    # Preprocessor config
    num_preprocessors: int = 4,
    preprocessor_batch_size: int = 8,
    preprocessor_num_cpus: float = 1.0,
    # Detector config
    num_detectors: int = 1,
    detector_batch_size: int = 4,
    detector_num_gpus: float = 1.0,
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
    # Server config
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    Start the full instrument detection service.

    This starts:
    1. Ray cluster (if not already running)
    2. FileUploader Ray Serve deployment
    3. Pipeline consumer in background thread
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    logger.info("Ray initialized")

    # Create shared job queue
    job_queue = Queue(maxsize=queue_max_size)
    logger.info(f"Job queue created (max_size={queue_max_size})")

    # Create the pipeline
    pipeline = create_pipeline(
        job_queue=job_queue,
        num_preprocessors=num_preprocessors,
        preprocessor_batch_size=preprocessor_batch_size,
        preprocessor_num_cpus=preprocessor_num_cpus,
        num_detectors=num_detectors,
        detector_batch_size=detector_batch_size,
        detector_num_gpus=detector_num_gpus,
        model_name=model_name,
    )
    logger.info("Pipeline created")

    # Setup shutdown handling
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"Received {sig_name}, initiating shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start pipeline consumer in background thread
    consumer_thread = threading.Thread(
        target=run_pipeline_consumer,
        args=(pipeline, None, shutdown_event),
        daemon=True,
    )
    consumer_thread.start()
    logger.info("Pipeline consumer started in background")

    # Deploy FileUploader with Ray Serve
    logger.info("Deploying FileUploader...")

    # Reconfigure the deployment with runtime options
    uploader_deployment = FileUploader.options(
        num_replicas=num_uploader_replicas,
    ).bind(
        job_queue=job_queue,
        max_file_size_mb=max_file_size_mb,
    )

    serve.run(
        uploader_deployment,
        name="instrument-detection",
        route_prefix="/",
        host=host,
        port=port,
    )

    logger.info(f"FileUploader deployed at http://{host}:{port}")
    logger.info("Service is ready. Press Ctrl+C to shutdown.")

    # Wait for shutdown signal
    try:
        while not shutdown_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    # Cleanup
    logger.info("Shutting down...")
    shutdown_event.set()

    # Stop the pipeline
    pipeline.stop()

    # Wait for consumer thread
    consumer_thread.join(timeout=5.0)

    # Shutdown Serve
    serve.shutdown()

    # Shutdown Ray
    ray.shutdown()

    logger.info("Shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Instrument Detection Service")

    # Queue config
    parser.add_argument(
        "--queue-size", type=int, default=1000, help="Max job queue size"
    )

    # FileUploader config
    parser.add_argument(
        "--num-uploaders", type=int, default=2, help="Number of uploader replicas"
    )
    parser.add_argument(
        "--max-file-size", type=float, default=100.0, help="Max file size in MB"
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
        "--model", type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help="Model name"
    )

    # Server config
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    args = parser.parse_args()

    print(
        f"""
Instrument Detection Service Configuration:
  Queue: max_size={args.queue_size}
  FileUploader: replicas={args.num_uploaders}, max_file_size={args.max_file_size}MB
  Preprocessor: actors={args.num_preprocessors}, batch={args.prep_batch}, cpus={args.prep_cpus}
  Detector: actors={args.num_detectors}, batch={args.detector_batch}, gpus={args.detector_gpus}
  Model: {args.model}
  Server: {args.host}:{args.port}
"""
    )

    main(
        queue_max_size=args.queue_size,
        num_uploader_replicas=args.num_uploaders,
        max_file_size_mb=args.max_file_size,
        num_preprocessors=args.num_preprocessors,
        preprocessor_batch_size=args.prep_batch,
        preprocessor_num_cpus=args.prep_cpus,
        num_detectors=args.num_detectors,
        detector_batch_size=args.detector_batch,
        detector_num_gpus=args.detector_gpus,
        model_name=args.model,
        host=args.host,
        port=args.port,
    )
