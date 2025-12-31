from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class InstrumentDetectJob:
    """Job containing audio data for instrument detection."""

    job_id: str
    created_at: int
    song_id: str
    song_hash: str
    filename: str
    audio_bytes: Optional[bytes] = None  # Raw audio bytes (preferred for streaming)


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
    """Result from basic instrument detection."""

    job_id: str
    song_id: str
    song_hash: str
    filename: str
    instruments: List[str]
    detected_at: int


@dataclass
class InstrumentDetectCoTResult:
    """Result from chain-of-thought instrument detection with layer information."""

    job_id: str
    song_id: str
    song_hash: str
    filename: str
    background: List[str] = field(default_factory=list)
    middle_ground: List[str] = field(default_factory=list)
    foreground: List[str] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)  # flattened
    detected_at: int = 0
