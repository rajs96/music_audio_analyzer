from dataclasses import dataclass
from ray._raylet import ObjectRef


@dataclass
class InstrumentDetectJob:
    job_id: str
    created_at: int
    song_id: str
    song_hash: str
    audio_ref: ObjectRef
    filename: str
