from dataclasses import dataclass
from ray._raylet import ObjectRef
import numpy as np


@dataclass
class InstrumentDetectJob:
    job_id: str
    created_at: int
    song_id: str
    song_hash: str
    audio_ref: ObjectRef
    filename: str


@dataclass
class PreprocessedAudio:
    """Intermediate data between preprocessor and detector."""

    job_id: str
    song_id: str
    song_hash: str
    filename: str
    waveform: np.ndarray


@dataclass
class InstrumentDetectResult:
    job_id: str
    song_id: str
    song_hash: str
    filename: str
    instruments: list[str]
    detected_at: int
