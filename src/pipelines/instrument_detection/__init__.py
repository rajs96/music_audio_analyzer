"""
Instrument Detection Pipeline.

A streaming pipeline for detecting instruments in audio files.

Usage (Streaming Pipeline):
    from ray.util.queue import Queue
    from src.pipelines.instrument_detection import (
        create_pipeline,
        result_from_row,
    )

    job_queue = Queue(maxsize=1000)
    pipeline = create_pipeline(job_queue)

    for batch in pipeline.stream():
        for row in batch:
            result = result_from_row(row)
            print(result.instruments)

Usage (Production with Ray Serve):
    from ray import serve
    from ray.util.queue import Queue
    from src.pipelines.instrument_detection import (
        create_pipeline,
        FileUploader,
    )

    # Create shared job queue
    job_queue = Queue(maxsize=1000)

    # Create and deploy the file uploader
    uploader = FileUploader.bind(job_queue=job_queue)
    serve.run(uploader, name="file-uploader", route_prefix="/")

    # Create and run the pipeline (in separate process/thread)
    pipeline = create_pipeline(job_queue)
    for batch in pipeline.stream():
        process_results(batch)
"""

from .data_classes import (
    InstrumentDetectJob,
    InstrumentDetectResult,
    PreprocessedAudio,
)
from .agents import (
    AudioPreprocessorAgent,
    InstrumentDetectorAgent,
)
from .pipeline import (
    create_pipeline,
    job_to_row,
    result_from_row,
)
from .file_uploader import FileUploader

__all__ = [
    # Data classes
    "InstrumentDetectJob",
    "InstrumentDetectResult",
    "PreprocessedAudio",
    # Agents
    "AudioPreprocessorAgent",
    "InstrumentDetectorAgent",
    # Pipeline
    "create_pipeline",
    "job_to_row",
    "result_from_row",
    # Serve
    "FileUploader",
]
