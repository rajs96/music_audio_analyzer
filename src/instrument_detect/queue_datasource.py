import ray.data as rd
from ray.util.queue import Queue, Empty
from typing import List
from src.instrument_detect.data_classes import InstrumentDetectJob


class QueueDatasource(rd.datasource.Datasource):
    def __init__(self, queue: Queue, pull_n: int = 128):
        self.queue = queue
        self.pull_n = pull_n

    def get_read_tasks(self, parallelism: int):
        queue = self.queue
        pull_n = self.pull_n

        def make_reader():
            import time

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
                    time.sleep(0.01)  # idle wait when queue is empty
                    continue

                yield items

        return [
            rd.datasource.ReadTask(make_reader, metadata=None)
            for _ in range(parallelism)
        ]
