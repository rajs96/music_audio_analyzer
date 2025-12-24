import ray.data as rd
from ray.util.queue import Queue, Empty
from typing import List
from src.instrument_detect.data_classes import InstrumentDetectJob
import time


class QueueDatasource(rd.datasource.Datasource):
    def __init__(self, queue: Queue, pull_n: int = 128):
        self.queue = queue

    def get_read_tasks(
        self, parallelism: int, pull_n: int = 128, idle_wait: float = 0.01
    ):
        queue = self.queue

        def make_reader():
            while True:
                items: List[InstrumentDetectJob] = []

                # Batch get items without blocking remote calls
                for _ in range(pull_n):
                    try:
                        item = queue.get_nowait()
                        items.append(item)
                    except Empty:
                        break

                if not items:
                    time.sleep(idle_wait)  # idle wait when queue is empty
                    continue

                yield items

        return [
            rd.datasource.ReadTask(make_reader, metadata=None)
            for _ in range(parallelism)
        ]
