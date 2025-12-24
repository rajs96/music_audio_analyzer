import ray
from ray.util.queue import Queue


def create_job_queue(
    max_size: int = 1000, name: str = "instrument_detect_queue"
) -> Queue:
    """Create a Ray Queue for InstrumentDetectJob objects."""
    return Queue(maxsize=max_size, actor_options={"name": name})


def get_existing_queue(name: str = "instrument_detect_queue") -> Queue:
    """Get an existing named queue."""
    actor = ray.get_actor(name)
    return Queue(actor=actor)


@ray.remote
class QueueMonitor:
    """Helper actor for debugging/inspecting the queue contents."""

    def __init__(self, queue: Queue):
        self.queue = queue

    def size(self) -> int:
        return self.queue.qsize()

    def empty(self) -> bool:
        return self.queue.empty()

    def full(self) -> bool:
        return self.queue.full()
