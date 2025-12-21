import ray
from loguru import logger
from src.instrument_detect.data_classes import InstrumentDetectJob


@ray.remote
class InstrumentDetectJobQueue:
    def __init__(self, max_size: int = 1000):
        self.q = []
        self.max_size = max_size

    def enqueue(self, job: InstrumentDetectJob):
        if len(self.q) < self.max_size:
            self.q.append(job)

    def dequeue_many(self, n: int) -> list[InstrumentDetectJob]:
        if not self.q:
            return []
        out = self.q[:n]
        del self.q[:n]
        return out
