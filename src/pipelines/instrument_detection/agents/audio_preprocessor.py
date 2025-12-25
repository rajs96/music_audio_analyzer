"""
Audio Preprocessor Agent.

Decodes audio bytes to waveforms for downstream processing.
"""

import tempfile
from typing import Any, Dict, List

import numpy as np
import ray
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
            "audio_ref": ObjectRef,  # Ray ObjectRef to audio bytes
        }

    Output format:
        {
            "job_id": str,
            "song_id": str,
            "song_hash": str,
            "filename": str,
            "waveform_ref": ObjectRef,  # Ray ObjectRef to decoded waveform (memory efficient)
            "error": Optional[str],  # Error message if preprocessing failed
        }
    """

    def __init__(self, target_sr: int = 16000, use_object_store: bool = True):
        super().__init__()
        self.target_sr = target_sr
        self.use_object_store = use_object_store  # Store waveforms in Ray object store

    def setup(self) -> None:
        """Initialize preprocessing resources."""
        logger.info(f"AudioPreprocessorAgent setup (target_sr={self.target_sr})")

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
        results = []

        for item in items:
            job_id = item.get("job_id", "unknown")
            song_id = item.get("song_id", "unknown")
            song_hash = item.get("song_hash", "unknown")
            filename = item.get("filename", "unknown")

            try:
                audio_ref = item["audio_ref"]

                # Get suffix for decoding
                suffix = filename.split(".")[-1] if "." in filename else "mp3"

                # Fetch audio bytes from Ray object store
                audio_bytes = ray.get(audio_ref)

                # Decode to waveform
                waveform = self.decode_audio_bytes_to_waveform(audio_bytes, suffix)

                # Store waveform in Ray object store for memory efficiency
                if self.use_object_store:
                    waveform_ref = ray.put(waveform)
                    results.append(
                        {
                            "job_id": job_id,
                            "song_id": song_id,
                            "song_hash": song_hash,
                            "filename": filename,
                            "waveform_ref": waveform_ref,
                            "error": None,
                        }
                    )
                else:
                    # Keep waveform inline (for testing or small batches)
                    results.append(
                        {
                            "job_id": job_id,
                            "song_id": song_id,
                            "song_hash": song_hash,
                            "filename": filename,
                            "waveform": waveform,
                            "error": None,
                        }
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
                        "waveform_ref": None,
                        "error": error_msg,
                    }
                )

        return results
