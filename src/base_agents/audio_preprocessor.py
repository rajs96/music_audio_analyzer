"""
Audio Preprocessor Agent.

Decodes audio bytes to waveforms for downstream processing.

This agent receives audio bytes and outputs waveforms serialized as bytes.
Downstream stages deserialize with np.frombuffer().
"""

import tempfile
from typing import Any, Dict, List

import numpy as np
import torchaudio
from loguru import logger

from src.streaming_pipeline import Agent


class AudioPreprocessorAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Agent that decodes audio bytes to waveforms.

    Input format:
        {
            "job_id": str,
            "song_id": str,
            "song_hash": str,
            "filename": str,
            "audio_bytes": bytes,  # Raw audio bytes (mp3, wav, etc.)
        }

    Output format:
        {
            "job_id": str,
            "song_id": str,
            "song_hash": str,
            "filename": str,
            "waveform_bytes": bytes,  # Waveform serialized as bytes (use np.frombuffer to decode)
            "sample_rate": int,  # Sample rate of the waveform
            "duration_seconds": float,  # Duration in seconds
            "error": str,  # Error message if failed, empty string "" if success
        }
    """

    def __init__(self, target_sr: int = 16000):
        super().__init__()
        self.target_sr = target_sr

    def setup(self) -> None:
        logger.info(f"AudioPreprocessorAgent setup (target_sr={self.target_sr})")
        import torchaudio  # warm up

        logger.info("AudioPreprocessorAgent ready")

    def decode_audio_bytes_to_waveform(
        self, audio_bytes: bytes, suffix: str
    ) -> np.ndarray:
        """Decode audio bytes to waveform at target sample rate."""
        with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=True) as f:
            f.write(audio_bytes)
            f.flush()
            wav, sr = torchaudio.load(f.name)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.target_sr
            )

        return wav.squeeze(0).numpy().astype("float32")

    def process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of audio jobs into preprocessed waveforms."""
        logger.debug(f"AudioPreprocessor processing batch of {len(items)} items")
        results = []

        for idx, item in enumerate(items):
            job_id = item.get("job_id", "unknown")
            song_id = item.get("song_id", "unknown")
            song_hash = item.get("song_hash", "unknown")
            filename = item.get("filename", "unknown")

            try:
                audio_bytes = item.get("audio_bytes")
                if audio_bytes is None:
                    raise ValueError("audio_bytes is None - missing audio data")

                suffix = filename.split(".")[-1] if "." in filename else "mp3"
                waveform = self.decode_audio_bytes_to_waveform(audio_bytes, suffix)
                duration_seconds = len(waveform) / self.target_sr
                waveform_bytes = waveform.tobytes()

                results.append(
                    {
                        "job_id": job_id,
                        "song_id": song_id,
                        "song_hash": song_hash,
                        "filename": filename,
                        "waveform_bytes": waveform_bytes,
                        "sample_rate": self.target_sr,
                        "duration_seconds": duration_seconds,
                        "error": "",
                    }
                )

                logger.debug(
                    f"Decoded {filename}: {duration_seconds:.1f}s, "
                    f"{len(waveform)} samples, {len(waveform_bytes)} bytes"
                )

            except Exception as e:
                error_msg = f"Preprocessing failed: {str(e)}"
                logger.error(f"Failed to preprocess {filename}: {e}")
                results.append(
                    {
                        "job_id": job_id,
                        "song_id": song_id,
                        "song_hash": song_hash,
                        "filename": filename,
                        "waveform_bytes": b"",
                        "sample_rate": None,
                        "duration_seconds": None,
                        "error": error_msg,
                    }
                )

        logger.debug(f"AudioPreprocessor completed batch of {len(results)} items")
        return results
