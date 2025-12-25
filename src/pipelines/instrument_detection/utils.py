import uuid
import time
import hashlib
import ray
from .data_classes import InstrumentDetectJob
from fastapi import UploadFile


def files_to_detection_jobs(files: list[UploadFile]) -> list[InstrumentDetectJob]:
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    created_at = int(time.time())

    detection_jobs = []
    for f in files:
        audio_bytes = f.read()
        song_id = f"trk_{uuid.uuid4().hex[:12]}"
        song_hash = hashlib.sha256(audio_bytes).hexdigest()

        audio_ref = ray.put(
            audio_bytes
        )  # store in ray object store of deployment, later s3

        detect_job = InstrumentDetectJob(
            job_id=job_id,
            created_at=created_at,
            song_id=song_id,
            song_hash=song_hash,
            audio_ref=audio_ref,
            filename=f.filename,
        )
        detection_jobs.append(detect_job)

    return detection_jobs
