"""
StreamingDatasource: A Ray Data Datasource for continuous streaming.

This module provides base classes for creating datasources that stream data
continuously (infinite generators) rather than terminating when the source
is temporarily empty.

Key insight from Ray Data source code (datasource.py:385-405):
    ReadTask.__call__() is a generator that yields blocks. For continuous
    streaming, this generator should never terminate until explicitly signaled.
"""

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar
import threading
import time

import ray
import ray.data as rd
from ray.data.block import Block
from loguru import logger


def _serialize_row_for_pyarrow(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize a row dict for safe PyArrow table creation.

    Handles special types like Ray ObjectRef that PyArrow can't serialize directly.
    These are pickled and stored as bytes, then unpickled when read.
    """
    import numpy as np

    serialized = {}
    for key, value in row.items():
        if isinstance(value, ray._raylet.ObjectRef):
            # Pickle ObjectRef and store as bytes with a marker prefix
            serialized[key] = b"__PICKLED_OBJREF__" + pickle.dumps(value)
        elif isinstance(value, np.ndarray):
            # Keep numpy arrays as-is, PyArrow handles them
            serialized[key] = value
        elif not isinstance(value, (str, int, float, bool, bytes, type(None), list)):
            # For other complex types, try pickling
            try:
                serialized[key] = b"__PICKLED__" + pickle.dumps(value)
            except Exception:
                # If pickling fails, convert to string
                serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def _deserialize_row_from_pyarrow(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize a row dict that was serialized for PyArrow.

    Unpickles values that were pickled during serialization.
    """
    deserialized = {}
    for key, value in row.items():
        if isinstance(value, bytes):
            if value.startswith(b"__PICKLED_OBJREF__"):
                deserialized[key] = pickle.loads(value[len(b"__PICKLED_OBJREF__") :])
            elif value.startswith(b"__PICKLED__"):
                deserialized[key] = pickle.loads(value[len(b"__PICKLED__") :])
            else:
                deserialized[key] = value
        else:
            deserialized[key] = value
    return deserialized


# Sentinel value to signal shutdown (use a unique string for serialization compatibility)
STOP_SENTINEL = "__STREAMING_DATASOURCE_STOP_SENTINEL__"

T = TypeVar("T")  # Input item type


@dataclass
class StreamingDatasourceConfig:
    """Configuration for StreamingDatasource behavior."""

    # Number of parallel read tasks (readers pulling from source)
    parallelism: int = 1

    # Number of items to batch before yielding a block
    # Keep small for faster streaming start
    batch_size: int = 8

    # How long to wait (seconds) before yielding a partial batch
    # Yield quickly to enable streaming - don't wait for full batch
    batch_timeout: float = 0.5

    # How long to wait (seconds) when source is empty before checking again
    poll_interval: float = 0.05

    # Optional: maximum items to read (None = infinite)
    max_items: Optional[int] = None


class StreamingDatasource(rd.datasource.Datasource, ABC, Generic[T]):
    """
    Abstract base class for datasources that stream continuously.

    Unlike standard Ray Data datasources that terminate when input is exhausted,
    StreamingDatasource yields blocks indefinitely until explicitly stopped.

    Subclasses must implement:
        - get_next_item(): Fetch the next item from the source (blocking or non-blocking)
        - item_to_row(): Convert an item to a dictionary row for the block

    Example usage:
        class QueueStreamingDatasource(StreamingDatasource[MyJob]):
            def __init__(self, queue: Queue, config: StreamingDatasourceConfig):
                super().__init__(config)
                self.queue = queue

            def get_next_item(self, timeout: float) -> Optional[MyJob]:
                try:
                    return self.queue.get(timeout=timeout)
                except Empty:
                    return None

            def item_to_row(self, item: MyJob) -> Dict[str, Any]:
                return {"job_id": item.job_id, "data": item.data}

        # Create dataset that streams forever
        ds = ray.data.read_datasource(
            QueueStreamingDatasource(my_queue, config)
        )

        # Iterate - will block waiting for new items
        for batch in ds.iter_batches():
            process(batch)
    """

    def __init__(self, config: Optional[StreamingDatasourceConfig] = None):
        super().__init__()  # Initialize base Datasource and mixins
        self.config = config or StreamingDatasourceConfig()
        self._stop_event = threading.Event()
        self._items_read = 0
        self._items_read_lock = threading.Lock()  # Thread-safe counter

    @abstractmethod
    def get_next_item(self, timeout: float) -> Optional[T]:
        """
        Fetch the next item from the source.

        This method should:
        - Block for up to `timeout` seconds waiting for an item
        - Return None if no item is available within the timeout
        - Return the item if available
        - Return STOP_SENTINEL to signal clean shutdown

        Args:
            timeout: Maximum seconds to wait for an item

        Returns:
            The next item, None if timeout, or STOP_SENTINEL to stop
        """
        pass

    @abstractmethod
    def item_to_row(self, item: T) -> Dict[str, Any]:
        """
        Convert an item to a dictionary row for the block.

        Args:
            item: The item fetched from the source

        Returns:
            Dictionary representing a single row in the output block
        """
        pass

    def stop(self):
        """Signal the datasource to stop streaming."""
        logger.info("StreamingDatasource stop requested")
        self._stop_event.set()

    def is_stopped(self) -> bool:
        """Check if stop has been requested."""
        return self._stop_event.is_set()

    def estimate_inmemory_data_size(self) -> Optional[int]:
        """Return estimated in-memory size, None for streaming (unknown)."""
        return None

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: Optional[int] = None,
        data_context=None,
    ) -> List["rd.datasource.ReadTask"]:
        """
        Create read tasks for Ray Data.

        Each read task is an infinite generator that yields blocks
        until stop() is called.
        """
        # Force parallelism=1 for queue-based streaming to avoid multiple readers
        # competing for the same queue items
        actual_parallelism = 1

        # Extract config values to avoid capturing self in closure
        # Only capture serializable primitives and Ray actor handles
        batch_size = self.config.batch_size
        batch_timeout = self.config.batch_timeout
        poll_interval = self.config.poll_interval
        max_items = self.config.max_items
        # Ray Queue is an actor handle and should be serializable
        queue = getattr(self, "queue", None)

        def make_block_generator() -> Iterable[Block]:
            """
            Infinite generator that yields blocks of rows.

            Uses STOP_SENTINEL in queue for shutdown signaling (no threading.Event).
            """
            import pickle
            import numpy as np
            import pyarrow as pa
            import ray
            from ray.util.queue import Empty
            from loguru import (
                logger,
            )  # Import inside function to avoid closure serialization issues

            # Define serialization inline to avoid module import issues on Ray workers
            def serialize_row(row):
                """Serialize a row dict for safe PyArrow table creation."""
                serialized = {}
                for key, value in row.items():
                    if isinstance(value, ray._raylet.ObjectRef):
                        serialized[key] = b"__PICKLED_OBJREF__" + pickle.dumps(value)
                    elif isinstance(value, np.ndarray):
                        serialized[key] = value
                    elif not isinstance(
                        value, (str, int, float, bool, bytes, type(None), list)
                    ):
                        try:
                            serialized[key] = b"__PICKLED__" + pickle.dumps(value)
                        except Exception:
                            serialized[key] = str(value)
                    else:
                        serialized[key] = value
                return serialized

            logger.info(f"Generator started. queue={queue}, max_items={max_items}")

            if queue is None:
                logger.error("Queue is None! Cannot read items.")
                return

            items_read = 0
            stopped = False
            last_log_time = time.time()

            while not stopped:
                # Check max items limit
                if max_items is not None and items_read >= max_items:
                    logger.info(f"Reached max_items limit ({max_items})")
                    break

                # Collect a batch of items
                batch_rows: List[Dict[str, Any]] = []
                batch_deadline = time.time() + batch_timeout

                while len(batch_rows) < batch_size:
                    if stopped:
                        break

                    # Calculate remaining time for this batch
                    remaining = batch_deadline - time.time()
                    if remaining <= 0:
                        break  # Batch timeout reached

                    # Try to get next item from queue
                    timeout = min(remaining, poll_interval)
                    try:
                        item = queue.get(timeout=timeout)
                        if items_read < 5 or items_read % 100 == 0:
                            logger.debug(f"Got item #{items_read}")
                    except Empty:
                        # Periodic logging when waiting
                        if time.time() - last_log_time > 5.0:
                            logger.debug(
                                f"Waiting for items... items_read={items_read}, batch_rows={len(batch_rows)}"
                            )
                            last_log_time = time.time()
                        continue

                    if item == "__STREAMING_DATASOURCE_STOP_SENTINEL__":
                        # Clean shutdown signal - but first drain remaining items
                        logger.info(
                            "Received STOP_SENTINEL, draining remaining items..."
                        )
                        from ray.util.queue import Empty as QueueEmpty

                        while True:
                            try:
                                remaining_item = queue.get(timeout=0.1)
                                if (
                                    remaining_item
                                    == "__STREAMING_DATASOURCE_STOP_SENTINEL__"
                                ):
                                    continue
                                if isinstance(remaining_item, dict):
                                    row = remaining_item
                                else:
                                    row = {"data": remaining_item}
                                serialized_row = serialize_row(row)
                                batch_rows.append(serialized_row)
                                items_read += 1
                            except QueueEmpty:
                                break
                        logger.info(f"Drained queue, total items read: {items_read}")
                        stopped = True
                        break

                    # Convert item to row - items should already be dicts
                    try:
                        if isinstance(item, dict):
                            row = item
                        else:
                            row = {"data": item}
                        # Serialize row for safe PyArrow table creation
                        serialized_row = serialize_row(row)
                        batch_rows.append(serialized_row)
                        items_read += 1
                    except Exception as e:
                        logger.error(f"Error converting item to row: {e}")
                        continue

                # Yield batch as a PyArrow table if we have items
                if batch_rows:
                    try:
                        table = pa.Table.from_pylist(batch_rows)
                        logger.debug(
                            f"Yielding batch with {len(batch_rows)} rows, total_read={items_read}"
                        )
                        yield table
                    except Exception as e:
                        logger.error(f"Error creating PyArrow table: {e}")
                        continue
                elif not stopped:
                    # Empty batch but not stopping - brief sleep to avoid busy loop
                    time.sleep(poll_interval)

            logger.info(f"StreamingDatasource reader exiting, read {items_read} items")

        # Create read tasks with minimal metadata for streaming
        from ray.data.block import BlockMetadata
        from ray.data.datasource import ReadTask

        metadata = BlockMetadata(
            num_rows=None,
            size_bytes=None,
            input_files=None,
            exec_stats=None,
        )

        return [
            ReadTask(
                read_fn=make_block_generator,
                metadata=metadata,
                per_task_row_limit=per_task_row_limit,
            )
            for _ in range(actual_parallelism)
        ]


class QueueStreamingDatasource(StreamingDatasource[T]):
    """
    A StreamingDatasource that reads from a Ray Queue.

    This is a concrete implementation for the common case of
    reading from a Ray distributed queue.

    For production use with concurrent producer/consumer:
    - Set expected_items to the number of items the producer will submit
    - The reader will stop after reading expected_items (no STOP_SENTINEL needed)
    - Or use max_items in config for the same effect

    Example:
        from ray.util.queue import Queue

        job_queue = Queue()
        datasource = QueueStreamingDatasource(
            queue=job_queue,
            item_to_row_fn=lambda job: {"id": job.id, "data": job.data},
            config=StreamingDatasourceConfig(batch_size=64, max_items=1000)
        )

        ds = ray.data.read_datasource(datasource)
    """

    def __init__(
        self,
        queue: "ray.util.queue.Queue",
        item_to_row_fn: Callable[[T], Dict[str, Any]],
        config: Optional[StreamingDatasourceConfig] = None,
        expected_items: Optional[int] = None,
    ):
        super().__init__(config)
        self.queue = queue
        self._item_to_row_fn = item_to_row_fn
        # If expected_items is set, use it as max_items
        if expected_items is not None:
            self.config.max_items = expected_items

    def get_next_item(self, timeout: float) -> Optional[T]:
        """Fetch next item from Ray Queue."""
        from ray.util.queue import Empty

        try:
            item = self.queue.get(timeout=timeout)

            # Check for sentinel
            if item == STOP_SENTINEL:
                return STOP_SENTINEL

            return item
        except Empty:
            return None

    def item_to_row(self, item: T) -> Dict[str, Any]:
        """Convert item using the provided function."""
        return self._item_to_row_fn(item)

    def signal_stop_via_queue(self):
        """
        Signal stop by pushing STOP_SENTINEL to the queue.

        Note: For concurrent producer/consumer, prefer using expected_items
        or max_items instead of STOP_SENTINEL to avoid race conditions.
        """
        self.queue.put(STOP_SENTINEL)
