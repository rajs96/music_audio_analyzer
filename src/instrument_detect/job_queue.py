import ray
from loguru import logger
from src.instrument_detect.data_classes import InstrumentDetectJob


@ray.remote
class InstrumentDetectJobQueue:
    def __init__(self, max_size: int = 1000):
        self.q: list[InstrumentDetectJob] = []
        self.max_size = max_size

    def enqueue(self, job: InstrumentDetectJob):
        if len(self.q) < self.max_size:
            self.q.append(job)
            return {"ok": True}
        return {"ok": False, "error": "queue_full"}

    def dequeue_many(self, n: int) -> list[InstrumentDetectJob]:
        if not self.q:
            return []
        out = self.q[:n]
        del self.q[:n]
        return out

    # -------- DEBUG / INSPECTION --------

    def size(self) -> int:
        return len(self.q)

    def peek(self, n: int = 5):
        """
        Return metadata for first n jobs without removing them.
        DO NOT return audio_ref contents.
        """
        return [
            {
                "job_id": j.job_id,
                "song_id": j.song_id,
                "filename": j.filename,
                "song_hash": j.song_hash,
                "created_at": j.created_at,
            }
            for j in self.q[:n]
        ]
