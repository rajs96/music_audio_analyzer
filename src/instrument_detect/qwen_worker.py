import time
from typing import List, Dict, Any
import ray
from src.instrument_detect.queue import Queue


@ray.remote(num_gpus=1)
class QwenWorker:
    def __init__(
        self,
        queue: Queue,
        result_store,
        batch_size: int = 8,
        max_wait_ms: int = 25,
    ):
        self.queue = queue
        self.result_store = result_store
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms

        # TODO: load Qwen model ONCE here (hot model)
        # self.model = load_qwen(...)
        # self.model.eval()

    def _run_qwen_batch(self, audio_bytes_list: List[bytes]) -> List[Dict[str, Any]]:
        pass

    def run_forever(self):
        while True:
            # 1) pull an initial batch (blocks up to a little)
            batch: List[Dict[str, Any]] = ray.get(
                self.detect_queue.dequeue_batch.remote(
                    max_items=self.batch_size, timeout_s=1.0
                )
            )
            if not batch:
                continue

            # 2) micro-batch: try to top up quickly for throughput
            deadline = time.time() + (self.max_wait_ms / 1000.0)
            while len(batch) < self.batch_size and time.time() < deadline:
                remaining = self.batch_size - len(batch)
                extra = ray.get(
                    self.detect_queue.dequeue_batch.remote(
                        max_items=remaining, timeout_s=0.0  # non-blocking
                    )
                )
                if not extra:
                    break
                batch.extend(extra)

            # 3) resolve audio refs concurrently
            audio_refs = [j["audio_ref"] for j in batch]
            audio_bytes_list: List[bytes] = ray.get(audio_refs)

            # 4) run true batched inference
            results = self._run_qwen_batch(audio_bytes_list)

            # 5) bulk write results (one RPC per batch if possible)
            # Best: implement result_store.put_detection_batch([...])
            payload = []
            now = int(time.time())
            for job, res in zip(batch, results):
                payload.append(
                    {
                        "job_id": job["job_id"],
                        "song_id": job["song_id"],
                        "song_hash": job["song_hash"],
                        "detected_at": now,
                        "result": res,
                    }
                )

            ray.get(self.result_store.put_detection_batch.remote(payload))
