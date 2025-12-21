from dataclasses import dataclass
import ray


@dataclass
class InstrumentDetectJob:
    job_id: str
    created_at: int
    song_id: str
    song_hash: str
    audio_ref: ray.ObjectRef
    filename: str
