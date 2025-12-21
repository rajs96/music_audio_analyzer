import ray
import ray.data as rd
from loguru import logger
from typing import List
from src.instrument_detect.data_classes import InstrumentDetectJob


class QueueDatasource(rd.datasource.Datasource):
    def __init__(self, queue_actor, pull_n=128):
        self.queue = queue_actor
        self.pull_n = pull_n

    def get_read_tasks(self, parallelism: int):
        # Create multiple read tasks so you can consume with multiple CPUs
        def make_reader():
            import time

            while True:
                items: List[InstrumentDetectJob] = ray.get(
                    self.queue.dequeue_many.remote(self.pull_n)
                )

                if not items:
                    time.sleep(0.01)  # idle wait
                    continue
                # Yield a "block" (e.g., a Python list of dicts)
                example_item = items[0] if items else None

                yield items

        return [
            rd.datasource.ReadTask(make_reader, metadata=None)
            for _ in range(parallelism)
        ]
