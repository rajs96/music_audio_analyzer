import time
from typing import List, Dict, Any
import ray
from ray.util.queue import Queue, Empty
from src.instrument_detect.data_classes import InstrumentDetectJob
from src.instrument_detect.instrument_detector.detector import InstrumentDetector


@ray.remote
class InstrumentDetectConsumer:
    def __init__(
        self,
        queue: Queue,
        worker: InstrumentDetector,
        batch_size: int = 8,
        max_wait_ms: int = 25,
    ):
        self.queue = queue
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms

        # TODO: load Qwen model ONCE here (hot model)
        # self.model = load_qwen(...)
        # self.model.eval()

    def _run_qwen_batch(self, audio_bytes_list: List[bytes]) -> List[Dict[str, Any]]:
        pass

    def _pull_batch(self, max_items: int) -> List[InstrumentDetectJob]:
        """Pull up to max_items from queue without blocking."""
        items = []
        for _ in range(max_items):
            try:
                item = self.queue.get_nowait()
                items.append(item)
            except Empty:
                break
        return items

    def run_forever(self):
        while True:
            # 1) pull an initial batch (non-blocking)
            batch = self._pull_batch(self.batch_size)
            if not batch:
                time.sleep(0.01)  # idle wait when queue is empty
                continue

            # 2) micro-batch: try to top up quickly for throughput
            deadline = time.time() + (self.max_wait_ms / 1000.0)
            while len(batch) < self.batch_size and time.time() < deadline:
                remaining = self.batch_size - len(batch)
                extra = self._pull_batch(remaining)
                if not extra:
                    break
                batch.extend(extra)

            # # 3) resolve audio refs concurrently
            # audio_refs = [j.audio_ref for j in batch]
            # audio_bytes_list: List[bytes] = ray.get(audio_refs)

            # # 4) run true batched inference
            # results = self._run_qwen_batch(audio_bytes_list)

            # 5) bulk write results (one RPC per batch if possible)
            # Best: implement result_store.put_detection_batch([...])
            payload = []
            now = int(time.time())
            # for job, res in zip(batch, results):
            #     payload.append(
            #         {
            #             "job_id": job.job_id,
            #             "song_id": job.song_id,
            #             "song_hash": job.song_hash,
            #             "detected_at": now,
            #             "result": res,
            #         }
            #     )
