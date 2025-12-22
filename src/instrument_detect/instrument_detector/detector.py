from abc import ABC, abstractmethod
from typing import List
import tempfile
import numpy as np
import torchaudio
import ray

from src.instrument_detect.data_classes import (
    InstrumentDetectJob,
    InstrumentDetectResult,
)
from src.instrument_detect.utils import load_model_and_processor


class InstrumentDetector(ABC):
    """
    Abstract base class for instrument detection models.

    All instrument detector implementations must provide both `process` and `predict` methods.
    """

    @abstractmethod
    def process(self, audio_bytes_list: List[bytes]) -> List[str]:
        pass

    @abstractmethod
    def predict(self, audio_bytes_list: List[bytes]) -> List[str]:
        """ """
        pass


class QwenInstrumentDetector(InstrumentDetector):
    """
    Instrument detector using Qwen3 Omni model.
    """

    def __init__(self):
        self.model_name = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
        self.model, self.processor = load_model_and_processor(self.model_name)

    def process(self, jobs: list[InstrumentDetectJob]) -> List[str]:
        waveform_audios = []
        for job in jobs:
            suffix = job.filename.split(".")[-1]
            audio_bytes = ray.get(job.audio_ref)
            waveform_audios.append(
                self.decode_audio_bytes_to_waveform(
                    audio_bytes, suffix, target_sr=16000
                )
            )

        return waveform_audios

    def predict(self, audio_ref_list) -> List[str]:
        pass

    def decode_audio_bytes_to_waveform(
        self, audio_bytes: bytes, suffix: str, target_sr: int = 16000
    ) -> np.ndarray:
        # Write bytes to a temp file so ffmpeg backend can decode it
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
            f.write(audio_bytes)
            f.flush()

            wav, sr = torchaudio.load(f.name)  # wav: (channels, time)

        # Convert to mono (optional, but typical)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to target_sr
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)

        return wav.squeeze(0).numpy().astype("float32")
