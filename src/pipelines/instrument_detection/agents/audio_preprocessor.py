"""
Audio Preprocessor Agent.

Decodes audio bytes to waveforms for downstream processing.

This agent receives audio bytes directly (not ObjectRefs) and outputs
waveforms as numpy arrays that can be serialized through Ray Data.
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
            "waveform": np.ndarray,  # Decoded waveform as numpy array (float32)
            "sample_rate": int,  # Sample rate of the waveform
            "duration_seconds": float,  # Duration in seconds
            "error": Optional[str],  # Error message if preprocessing failed
        }
    """

    def __init__(self, target_sr: int = 16000):
        super().__init__()
        self.target_sr = target_sr

    def setup(self) -> None:
        """Initialize preprocessing resources."""
        logger.info(f"AudioPreprocessorAgent setup (target_sr={self.target_sr})")
        # Pre-import torchaudio to warm up
        import torchaudio

        logger.info("AudioPreprocessorAgent ready")

    def decode_audio_bytes_to_waveform(
        self, audio_bytes: bytes, suffix: str
    ) -> np.ndarray:
        """Decode audio bytes to waveform at target sample rate."""
        with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=True) as f:
            f.write(audio_bytes)
            f.flush()
            wav, sr = torchaudio.load(f.name)

        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to target_sr
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
                # Get audio bytes directly from item
                audio_bytes = item.get("audio_bytes")

                if audio_bytes is None:
                    raise ValueError("audio_bytes is None - missing audio data")

                # Get suffix for decoding
                suffix = filename.split(".")[-1] if "." in filename else "mp3"

                # Decode to waveform
                waveform = self.decode_audio_bytes_to_waveform(audio_bytes, suffix)

                # Calculate duration
                duration_seconds = len(waveform) / self.target_sr

                results.append(
                    {
                        "job_id": job_id,
                        "song_id": song_id,
                        "song_hash": song_hash,
                        "filename": filename,
                        "waveform": waveform,
                        "sample_rate": self.target_sr,
                        "duration_seconds": duration_seconds,
                        "error": None,
                    }
                )

                logger.debug(
                    f"Decoded {filename}: {duration_seconds:.1f}s, "
                    f"{len(waveform)} samples"
                )

            except Exception as e:
                error_msg = f"Preprocessing failed: {str(e)}"
                logger.error(f"Failed to preprocess {filename}: {e}")

                # Propagate error instead of silently dropping
                results.append(
                    {
                        "job_id": job_id,
                        "song_id": song_id,
                        "song_hash": song_hash,
                        "filename": filename,
                        "waveform": None,
                        "sample_rate": None,
                        "duration_seconds": None,
                        "error": error_msg,
                    }
                )

        logger.debug(f"AudioPreprocessor completed batch of {len(results)} items")
        return results
