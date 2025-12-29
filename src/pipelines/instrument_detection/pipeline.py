"""
Instrument Detection Pipeline Configuration.

Creates a streaming pipeline for detecting instruments in audio files.
"""

from typing import Any, Dict
import torch
from ray.util.queue import Queue

from src.streaming_pipeline import (
    QueueStreamingDatasource,
    StreamingDatasourceConfig,
    AgentRayComputeConfig,
    AgentStage,
    StreamingPipeline,
)
from .data_classes import (
    InstrumentDetectJob,
    InstrumentDetectResult,
    InstrumentDetectCoTResult,
)
from .agents import (
    AudioPreprocessorAgent,
    InstrumentDetectorCoTAgent,
)

DEFAULT_COT_GENERATE_KWARGS = {
    "max_new_tokens": 512,
    "do_sample": False,
    "return_audio": False,
}

DEFAULT_GENERATE_KWARGS = {
    "max_new_tokens": 256,
    "do_sample": False,
    "return_audio": False,
}


def job_to_row(job: InstrumentDetectJob) -> Dict[str, Any]:
    """
    Convert an InstrumentDetectJob to a dictionary row for the datasource.

    Use this as the item_to_row_fn for QueueStreamingDatasource.
    """
    return {
        "job_id": job.job_id,
        "song_id": job.song_id,
        "song_hash": job.song_hash,
        "filename": job.filename,
        "audio_ref": job.audio_ref,
        "created_at": job.created_at,
    }


def result_from_row(row: Dict[str, Any]) -> InstrumentDetectResult:
    """
    Convert a result row back to an InstrumentDetectResult object.

    Use this to convert pipeline output back to the original data class.
    """
    return InstrumentDetectResult(
        job_id=row["job_id"],
        song_id=row["song_id"],
        song_hash=row["song_hash"],
        filename=row["filename"],
        instruments=row["instruments"],
        detected_at=row["detected_at"],
    )


def cot_result_from_row(row: Dict[str, Any]) -> InstrumentDetectCoTResult:
    """
    Convert a result row back to an InstrumentDetectCoTResult object.

    Use this to convert CoT pipeline output back to the original data class.
    """
    return InstrumentDetectCoTResult(
        job_id=row["job_id"],
        song_id=row["song_id"],
        song_hash=row["song_hash"],
        filename=row["filename"],
        background=row.get("background", []),
        middle_ground=row.get("middle_ground", []),
        foreground=row.get("foreground", []),
        instruments=row.get("instruments", []),
        detected_at=row["detected_at"],
    )


def create_pipeline(
    job_queue: Queue,
    # Datasource config
    datasource_batch_size: int = 32,
    datasource_parallelism: int = 1,
    # Preprocessor config
    num_preprocessors: int = 4,
    preprocessor_batch_size: int = 8,
    preprocessor_num_cpus: float = 1.0,
    # Detector config
    num_detectors: int = 1,
    detector_batch_size: int = 4,
    detector_num_gpus: float = 1.0,
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
) -> StreamingPipeline:
    """
    Create a streaming instrument detection pipeline.

    Args:
        job_queue: Ray Queue containing InstrumentDetectJob items
        datasource_batch_size: Batch size for pulling from queue
        datasource_parallelism: Number of parallel queue readers
        num_preprocessors: Number of preprocessor actors
        preprocessor_batch_size: Batch size for preprocessing
        preprocessor_num_cpus: CPUs per preprocessor actor
        num_detectors: Number of detector actors (usually 1 per GPU)
        detector_batch_size: Batch size for GPU inference
        detector_num_gpus: GPUs per detector actor
        model_name: Name of the model to use for detection

    Returns:
        StreamingPipeline ready to stream results
    """
    # Create the streaming datasource
    datasource = QueueStreamingDatasource(
        queue=job_queue,
        item_to_row_fn=job_to_row,
        config=StreamingDatasourceConfig(
            parallelism=datasource_parallelism,
            batch_size=datasource_batch_size,
            batch_timeout=0.1,
            poll_interval=0.01,
        ),
    )

    dtype = torch.bfloat16
    device = "cuda"
    # Create agent stages
    stages = [
        # Stage 1: Preprocessing (CPU-bound, parallelizable)
        AgentStage(
            agent=AudioPreprocessorAgent(target_sr=16000),
            config=AgentRayComputeConfig(
                num_actors=num_preprocessors,
                batch_size=preprocessor_batch_size,
                num_cpus=preprocessor_num_cpus,
                num_gpus=0,
            ),
            name="AudioPreprocessor",
        ),
        # Stage 2: Detection (GPU-bound)
        AgentStage(
            agent=InstrumentDetectorCoTAgent(
                model_name=model_name,
                dtype=dtype,
                device=device,
                planning_generate_kwargs=DEFAULT_COT_GENERATE_KWARGS,
                response_generate_kwargs=DEFAULT_GENERATE_KWARGS,
            ),
            config=AgentRayComputeConfig(
                num_actors=num_detectors,
                batch_size=detector_batch_size,
                num_cpus=1.0,
                num_gpus=detector_num_gpus,
            ),
            name="InstrumentDetector",
        ),
    ]

    # Create and return the pipeline
    return StreamingPipeline(
        datasource=datasource,
        stages=stages,
        name="InstrumentDetectPipeline",
    )


def create_cot_pipeline(
    job_queue: Queue,
    # Datasource config
    datasource_batch_size: int = 32,
    datasource_parallelism: int = 1,
    # Preprocessor config
    num_preprocessors: int = 4,
    preprocessor_batch_size: int = 8,
    preprocessor_num_cpus: float = 1.0,
    # Detector config
    num_detectors: int = 1,
    detector_batch_size: int = 4,
    detector_num_gpus: float = 1.0,
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
) -> StreamingPipeline:
    """
    Create a streaming instrument detection pipeline with chain-of-thought.

    Uses InstrumentDetectorCoTAgent for two-step reasoning:
    1. Describe sounds in background/middle-ground/foreground layers
    2. Convert descriptions to structured JSON with instruments

    Args:
        job_queue: Ray Queue containing InstrumentDetectJob items
        datasource_batch_size: Batch size for pulling from queue
        datasource_parallelism: Number of parallel queue readers
        num_preprocessors: Number of preprocessor actors
        preprocessor_batch_size: Batch size for preprocessing
        preprocessor_num_cpus: CPUs per preprocessor actor
        num_detectors: Number of detector actors (usually 1 per GPU)
        detector_batch_size: Batch size for GPU inference
        detector_num_gpus: GPUs per detector actor
        model_name: Name of the model to use for detection

    Returns:
        StreamingPipeline ready to stream results with layer-based instruments
    """
    # Create the streaming datasource
    datasource = QueueStreamingDatasource(
        queue=job_queue,
        item_to_row_fn=job_to_row,
        config=StreamingDatasourceConfig(
            parallelism=datasource_parallelism,
            batch_size=datasource_batch_size,
            batch_timeout=0.1,
            poll_interval=0.01,
        ),
    )

    # Create agent stages
    stages = [
        # Stage 1: Preprocessing (CPU-bound, parallelizable)
        AgentStage(
            agent=AudioPreprocessorAgent(target_sr=16000),
            config=AgentRayComputeConfig(
                num_actors=num_preprocessors,
                batch_size=preprocessor_batch_size,
                num_cpus=preprocessor_num_cpus,
                num_gpus=0,
            ),
            name="AudioPreprocessor",
        ),
        # Stage 2: CoT Detection (GPU-bound)
        AgentStage(
            agent=InstrumentDetectorCoTAgent(model_name=model_name),
            config=AgentRayComputeConfig(
                num_actors=num_detectors,
                batch_size=detector_batch_size,
                num_cpus=1.0,
                num_gpus=detector_num_gpus,
            ),
            name="InstrumentDetectorCoT",
        ),
    ]

    # Create and return the pipeline
    return StreamingPipeline(
        datasource=datasource,
        stages=stages,
        name="InstrumentDetectCoTPipeline",
    )
