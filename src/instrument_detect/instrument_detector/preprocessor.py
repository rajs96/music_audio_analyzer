import tempfile
import numpy as np
import torchaudio
from typing import List, Optional
import ray

from src.instrument_detect.data_classes import InstrumentDetectJob, PreprocessedAudio


class PreprocessorError(Exception):
    """Raised when preprocessor encounters a fatal error."""

    pass


@ray.remote
class PreprocessorActor:
    """
    Stateless preprocessor actor that decodes audio to waveforms.
    Designed to be used with ActorPool.

    Resource configuration is done at actor creation time via .options():
        PreprocessorActor.options(num_cpus=0.25, max_concurrency=4).remote()
    """

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self._error: Optional[str] = None
        self._stopped = False

    def is_healthy(self) -> bool:
        """Check if actor is healthy."""
        return self._error is None and not self._stopped

    def get_error(self) -> Optional[str]:
        """Get the error message if actor failed."""
        return self._error

    def stop(self):
        """Signal actor to stop."""
        self._stopped = True

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

    def process(self, job: InstrumentDetectJob) -> PreprocessedAudio:
        """Process a single job into preprocessed audio."""
        try:
            suffix = job.filename.split(".")[-1]
            audio_bytes = ray.get(job.audio_ref)
            waveform = self.decode_audio_bytes_to_waveform(audio_bytes, suffix)

            return PreprocessedAudio(
                job_id=job.job_id,
                song_id=job.song_id,
                song_hash=job.song_hash,
                filename=job.filename,
                waveform=waveform,
            )
        except Exception as e:
            from loguru import logger

            error_msg = f"Failed to process job {job.job_id}: {type(e).__name__}: {e}"
            logger.error(error_msg)
            self._error = error_msg
            raise PreprocessorError(error_msg) from e

    def process_batch(self, jobs: List[InstrumentDetectJob]) -> List[PreprocessedAudio]:
        """Process a batch of jobs."""
        return [self.process(job) for job in jobs]
