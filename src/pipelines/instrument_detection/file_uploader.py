"""
FileUploader: Ray Serve deployment for submitting audio files to the pipeline.

This class acts as the HTTP interface for users to submit audio files for
instrument detection. It receives file uploads, creates jobs, and puts them
into the streaming pipeline's input queue.

Usage:
    from ray import serve
    from ray.util.queue import Queue

    # Create the job queue
    job_queue = Queue(maxsize=1000)

    # Bind the queue to the deployment
    uploader = FileUploader.bind(job_queue=job_queue)

    # Deploy
    serve.run(uploader)

    # Now clients can POST files to the endpoint:
    # curl -X POST -F "file=@song.mp3" http://localhost:8000/upload
"""

import hashlib
import time
import uuid
from typing import Optional

import ray
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
from ray import serve
from ray.util.queue import Queue

from .data_classes import InstrumentDetectJob


# Response models
class UploadResponse(BaseModel):
    job_id: str
    song_id: str
    filename: str
    message: str


class HealthResponse(BaseModel):
    status: str
    queue_size: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# FastAPI app for the deployment
app = FastAPI(
    title="Instrument Detection File Uploader",
    description="Upload audio files for instrument detection",
    version="1.0.0",
)


@serve.deployment(
    name="file-uploader",
    num_replicas=2,
    ray_actor_options={"num_cpus": 0.5},
)
@serve.ingress(app)
class FileUploader:
    """
    Ray Serve deployment that accepts audio file uploads and submits them
    to the instrument detection pipeline.

    The job queue is bound at deployment time, allowing this class to be
    connected to the streaming pipeline's input queue.

    Attributes:
        job_queue: Ray Queue where jobs are submitted
        max_file_size_mb: Maximum allowed file size in MB
        allowed_extensions: Set of allowed file extensions
    """

    def __init__(
        self,
        job_queue: Queue,
        max_file_size_mb: float = 100.0,
        allowed_extensions: Optional[set] = None,
    ):
        """
        Initialize the FileUploader.

        Args:
            job_queue: Ray Queue to submit jobs to
            max_file_size_mb: Maximum file size allowed (default 100MB)
            allowed_extensions: Allowed file extensions (default: audio formats)
        """
        self.job_queue = job_queue
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.allowed_extensions = allowed_extensions or {
            "mp3",
            "wav",
            "flac",
            "ogg",
            "m4a",
            "aac",
            "wma",
            "aiff",
        }

        logger.info(
            f"FileUploader initialized: max_size={max_file_size_mb}MB, "
            f"extensions={self.allowed_extensions}"
        )

    def _validate_file(self, filename: str, file_size: int) -> None:
        """
        Validate the uploaded file.

        Raises:
            HTTPException: If validation fails
        """
        # Check extension
        if "." not in filename:
            raise HTTPException(
                status_code=400,
                detail=f"File must have an extension. Allowed: {self.allowed_extensions}",
            )

        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in self.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File extension '{ext}' not allowed. Allowed: {self.allowed_extensions}",
            )

        # Check size
        if file_size > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {max_mb:.1f}MB",
            )

    def _create_job(
        self,
        audio_bytes: bytes,
        filename: str,
        song_id: Optional[str] = None,
    ) -> InstrumentDetectJob:
        """
        Create an InstrumentDetectJob from uploaded audio bytes.

        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename
            song_id: Optional song ID (generated if not provided)

        Returns:
            InstrumentDetectJob ready to be queued
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        song_id = song_id or f"trk_{uuid.uuid4().hex[:12]}"
        song_hash = hashlib.sha256(audio_bytes).hexdigest()
        created_at = int(time.time())

        # Store audio bytes in Ray object store
        audio_ref = ray.put(audio_bytes)

        return InstrumentDetectJob(
            job_id=job_id,
            created_at=created_at,
            song_id=song_id,
            song_hash=song_hash,
            audio_ref=audio_ref,
            filename=filename,
        )

    @app.post(
        "/upload",
        response_model=UploadResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Invalid file"},
            413: {"model": ErrorResponse, "description": "File too large"},
            503: {"model": ErrorResponse, "description": "Queue full"},
        },
    )
    async def upload_file(
        self,
        file: UploadFile = File(..., description="Audio file to process"),
        song_id: Optional[str] = None,
    ) -> UploadResponse:
        """
        Upload an audio file for instrument detection.

        Args:
            file: The audio file to upload
            song_id: Optional identifier for the song

        Returns:
            UploadResponse with job details
        """
        # Read file content
        content = await file.read()
        file_size = len(content)

        logger.info(f"Received upload: {file.filename} ({file_size} bytes)")

        # Validate
        self._validate_file(file.filename, file_size)

        # Create job
        job = self._create_job(content, file.filename, song_id)

        # Submit to queue
        try:
            # Non-blocking put with timeout
            self.job_queue.put(job, block=True, timeout=5.0)
        except Exception as e:
            logger.error(f"Failed to queue job: {e}")
            raise HTTPException(
                status_code=503,
                detail="Queue is full. Please try again later.",
            )

        logger.info(f"Job queued: {job.job_id} for {job.filename}")

        return UploadResponse(
            job_id=job.job_id,
            song_id=job.song_id,
            filename=job.filename,
            message="File uploaded successfully. Processing will begin shortly.",
        )

    @app.post(
        "/upload/batch",
        response_model=list[UploadResponse],
        responses={
            400: {"model": ErrorResponse, "description": "Invalid file"},
            413: {"model": ErrorResponse, "description": "File too large"},
            503: {"model": ErrorResponse, "description": "Queue full"},
        },
    )
    async def upload_batch(
        self,
        files: list[UploadFile] = File(..., description="Audio files to process"),
    ) -> list[UploadResponse]:
        """
        Upload multiple audio files for instrument detection.

        Args:
            files: List of audio files to upload

        Returns:
            List of UploadResponse with job details for each file
        """
        responses = []

        for file in files:
            content = await file.read()
            file_size = len(content)

            logger.info(f"Received batch upload: {file.filename} ({file_size} bytes)")

            # Validate
            self._validate_file(file.filename, file_size)

            # Create job
            job = self._create_job(content, file.filename)

            # Submit to queue
            try:
                self.job_queue.put(job, block=True, timeout=5.0)
            except Exception as e:
                logger.error(f"Failed to queue job: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Queue is full after {len(responses)} files. Please try again later.",
                )

            logger.info(f"Job queued: {job.job_id} for {job.filename}")

            responses.append(
                UploadResponse(
                    job_id=job.job_id,
                    song_id=job.song_id,
                    filename=job.filename,
                    message="File uploaded successfully.",
                )
            )

        return responses

    @app.get("/health", response_model=HealthResponse)
    async def health_check(self) -> HealthResponse:
        """
        Check the health of the uploader and queue status.

        Returns:
            HealthResponse with status and queue size
        """
        try:
            queue_size = self.job_queue.qsize()
            return HealthResponse(status="healthy", queue_size=queue_size)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(status="unhealthy", queue_size=-1)

    @app.get("/")
    async def root(self) -> dict:
        """Root endpoint with API information."""
        return {
            "service": "Instrument Detection File Uploader",
            "version": "1.0.0",
            "endpoints": {
                "POST /upload": "Upload a single audio file",
                "POST /upload/batch": "Upload multiple audio files",
                "GET /health": "Health check and queue status",
            },
        }
