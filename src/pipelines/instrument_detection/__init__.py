"""
Instrument Detection Pipeline.

A streaming pipeline for detecting instruments in audio files.

Usage:
    from src.pipelines.instrument_detection import (
        create_pipeline,
        InstrumentDetectJob,
        InstrumentDetectResult,
        job_to_row,
        result_from_row,
    )

    pipeline = create_pipeline(job_queue)
    for batch in pipeline.stream():
        for row in batch:
            result = result_from_row(row)
            print(result.instruments)
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
]
