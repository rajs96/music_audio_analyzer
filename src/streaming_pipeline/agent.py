"""
Agent: Pure processing logic abstraction.

An Agent defines just the business logic for processing data.
The framework handles all Ray Data integration automatically.

This separation allows:
    - Agents to be simple, focused on processing logic
    - No boilerplate for batch format conversion
    - Agents can be tested independently of Ray Data
    - Same agent can be used in different contexts (Ray Data, direct calls, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from loguru import logger

# Type variables for input and output types
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


@dataclass
class AgentRayComputeConfig:
    """Configuration for Ray compute resources and scaling of an Agent.

    For vLLM with tensor/pipeline parallelism:
        - Set tensor_parallel_size and/or pipeline_parallel_size
        - num_gpus is calculated automatically as tp_size * pp_size
        - Use num_actors=1 (one vLLM engine sharded across GPUs)
        - Use distributed_executor_backend="mp" (default) or "ray" (cross-node)

    Example for 4-GPU tensor parallelism:
        config = AgentRayComputeConfig(
            num_actors=1,
            tensor_parallel_size=4,
            batch_size=8,
        )
    """

    # Number of actor replicas (for vLLM with TP/PP, use 1)
    num_actors: int = 1
    # Batch size for processing
    batch_size: int = 32
    # CPU resources per actor
    num_cpus: float = 1.0
    # GPU resources per actor (0 = no GPU)
    # NOTE: If tensor_parallel_size or pipeline_parallel_size > 1,
    # this is automatically set to tp_size * pp_size
    num_gpus: float = 0.0
    # Maximum concurrent tasks per actor
    max_concurrency: int = 1
    # Maximum time (ms) to wait for a full batch
    max_batch_wait_ms: int = 100

    # === vLLM Tensor/Pipeline Parallelism ===
    # Tensor parallelism: split model layers across GPUs
    tensor_parallel_size: int = 1
    # Pipeline parallelism: split model stages across GPUs
    pipeline_parallel_size: int = 1
    # Distributed executor backend: "mp" (multiprocessing) or "ray" (cross-node)
    distributed_executor_backend: str = "mp"
    # Accelerator type label (e.g., "A100", "H100") for scheduling
    accelerator_type: Optional[str] = None
    # Placement group strategy: "PACK", "STRICT_PACK", "SPREAD", "STRICT_SPREAD"
    placement_group_strategy: str = "PACK"

    def get_num_gpus_per_actor(self) -> float:
        """Calculate GPUs needed per actor based on parallelism settings."""
        parallelism_gpus = self.tensor_parallel_size * self.pipeline_parallel_size
        if parallelism_gpus > 1:
            return float(parallelism_gpus)
        return self.num_gpus

    def get_num_bundles_per_replica(self) -> int:
        """Get number of GPU bundles needed per replica for placement groups."""
        return self.tensor_parallel_size * self.pipeline_parallel_size


class Agent(ABC, Generic[TInput, TOutput]):
    """
    Abstract base class for processing agents.

    An Agent encapsulates the pure processing logic without any
    Ray Data specific code. The framework wraps agents automatically.

    Subclasses must implement:
        - process_batch(): Transform a list of inputs to outputs

    Optionally override:
        - setup(): One-time initialization (load models, etc.)
        - teardown(): Cleanup resources
        - process(): Single-item processing (if you prefer item-by-item)

    Example:
        class AudioDecoder(Agent[AudioJob, DecodedAudio]):
            def __init__(self, sample_rate: int = 16000):
                self.sample_rate = sample_rate

            def setup(self):
                # Load any resources
                pass

            def process_batch(self, items: List[AudioJob]) -> List[DecodedAudio]:
                return [self._decode(item) for item in items]

        # Use in pipeline - framework creates component automatically
        pipeline = StreamingPipeline(
            datasource=my_datasource,
            stages=[
                AgentStage(AudioDecoder(sample_rate=16000), AgentRayComputeConfig(num_actors=4)),
            ]
        )
    """

    def __init__(self):
        self._is_setup = False

    def setup(self) -> None:
        """
        One-time initialization called when actor starts.

        Override to load models, initialize connections, etc.
        Called once per actor replica.
        """
        pass

    def teardown(self) -> None:
        """
        Cleanup called when actor is shutting down.

        Override to release resources, close connections, etc.
        """
        pass

    def process(self, item: TInput) -> TOutput:
        """
        Process a single item.

        Override this for simple item-by-item processing.
        Default implementation calls process_batch with a single item.

        Args:
            item: Single input item

        Returns:
            Single output item
        """
        results = self.process_batch([item])
        return results[0] if results else None

    @abstractmethod
    def process_batch(self, items: List[TInput]) -> List[TOutput]:
        """
        Process a batch of items.

        This is the core method that subclasses must implement.

        Args:
            items: List of input items

        Returns:
            List of output items (should match input length for 1:1 mapping,
            or can be different length for filtering/expansion)
        """
        pass

    def _ensure_setup(self) -> None:
        """Ensure setup() has been called."""
        if not self._is_setup:
            self.setup()
            self._is_setup = True


class FunctionAgent(Agent[TInput, TOutput]):
    """
    An Agent that wraps a simple function.

    Use this for simple transformations without custom setup/teardown.

    Example:
        # Wrap a function as an agent
        agent = FunctionAgent(lambda items: [x * 2 for x in items])

        # Or with setup
        def my_setup():
            return load_model()

        def my_process(items, model):
            return [model.predict(x) for x in items]

        agent = FunctionAgent(
            process_fn=my_process,
            setup_fn=my_setup,
        )
    """

    def __init__(
        self,
        process_fn: Callable[[List[TInput]], List[TOutput]],
        setup_fn: Optional[Callable[[], Any]] = None,
        teardown_fn: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self._process_fn = process_fn
        self._setup_fn = setup_fn
        self._teardown_fn = teardown_fn
        self._setup_result = None

    def setup(self) -> None:
        if self._setup_fn:
            self._setup_result = self._setup_fn()

    def teardown(self) -> None:
        if self._teardown_fn:
            self._teardown_fn()

    def process_batch(self, items: List[TInput]) -> List[TOutput]:
        # If setup returned something (like a model), pass it to process_fn
        if self._setup_result is not None:
            return self._process_fn(items, self._setup_result)
        return self._process_fn(items)


# Type alias for input/output format specification
class BatchFormat:
    """Specifies how to convert between Ray Data batches and agent items."""

    DICT_OF_LISTS = "dict_of_lists"  # {"col1": [...], "col2": [...]}
    LIST_OF_DICTS = "list_of_dicts"  # [{"col1": v1, "col2": v2}, ...]
    RAW_ITEMS = "raw_items"  # Items stored in "item" column


@dataclass
class AgentStage:
    """
    A stage in a pipeline that wraps an Agent.

    This bundles an Agent with its deployment configuration.
    The pipeline uses this to create the appropriate Ray Data component.

    Example:
        stage = AgentStage(
            agent=MyProcessor(),
            config=AgentRayComputeConfig(num_actors=4, num_gpus=1),
            name="MyProcessor",
        )
    """

    agent: Agent
    config: AgentRayComputeConfig = None
    name: Optional[str] = None

    # Input/output format handling
    input_format: str = BatchFormat.LIST_OF_DICTS
    output_format: str = BatchFormat.LIST_OF_DICTS

    def __post_init__(self):
        if self.config is None:
            self.config = AgentRayComputeConfig()
        if self.name is None:
            self.name = self.agent.__class__.__name__


def create_agent_callable(
    agent_class: type,
    agent_args: tuple = (),
    agent_kwargs: dict = None,
    input_format: str = BatchFormat.LIST_OF_DICTS,
    output_format: str = BatchFormat.LIST_OF_DICTS,
) -> type:
    """
    Create a callable class that wraps an Agent for use with Ray Data.

    This is the bridge between Agents and Ray Data's map_batches.
    It handles all the batch format conversion automatically.

    Args:
        agent_class: The Agent subclass to wrap
        agent_args: Positional arguments for agent __init__
        agent_kwargs: Keyword arguments for agent __init__
        input_format: How to interpret incoming Ray Data batches
        output_format: How to format outgoing Ray Data batches

    Returns:
        A class suitable for use with ray.data.Dataset.map_batches()
    """
    if agent_kwargs is None:
        agent_kwargs = {}

    class AgentCallable:
        """Ray Data compatible wrapper for an Agent."""

        def __init__(self):
            from loguru import logger  # Import inside to avoid closure serialization

            self.agent = agent_class(*agent_args, **agent_kwargs)
            self.agent.setup()
            self.agent._is_setup = True
            logger.info(f"AgentCallable initialized: {agent_class.__name__}")

        def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            """Process a Ray Data batch through the agent."""
            # Convert Ray Data batch to list of items
            items = self._batch_to_items(batch, input_format)

            # Process through agent
            if items:
                outputs = self.agent.process_batch(items)
            else:
                outputs = []

            # Convert back to Ray Data batch
            return self._items_to_batch(outputs, output_format)

        def _batch_to_items(self, batch: Dict[str, Any], fmt: str) -> List[Any]:
            """Convert Ray Data batch (dict of columns) to list of items."""
            import pickle  # Local import for serialization safety

            def to_bytes_if_possible(value):
                """Convert value to bytes if it's a bytes-like type.

                PyArrow can return binary data as various types:
                - bytes (Python native)
                - numpy.bytes_
                - pyarrow.lib.LargeBinaryScalar / BinaryScalar

                This function normalizes them all to Python bytes.
                """
                if isinstance(value, bytes):
                    return value
                # Handle numpy bytes
                if hasattr(value, "tobytes"):
                    return value.tobytes()
                # Handle PyArrow scalars
                if hasattr(value, "as_py"):
                    py_val = value.as_py()
                    if isinstance(py_val, bytes):
                        return py_val
                # Handle memoryview
                if isinstance(value, memoryview):
                    return bytes(value)
                return None  # Not a bytes-like type

            def deserialize_row(row):
                """Deserialize pickled values from PyArrow storage."""
                deserialized = {}
                for key, value in row.items():
                    # Try to convert to bytes (handles PyArrow binary scalars, numpy bytes, etc.)
                    byte_value = to_bytes_if_possible(value)

                    if byte_value is not None:
                        if byte_value.startswith(b"__PICKLED_OBJREF__"):
                            deserialized[key] = pickle.loads(
                                byte_value[len(b"__PICKLED_OBJREF__") :]
                            )
                        elif byte_value.startswith(b"__PICKLED__"):
                            deserialized[key] = pickle.loads(
                                byte_value[len(b"__PICKLED__") :]
                            )
                        else:
                            deserialized[key] = byte_value
                    else:
                        deserialized[key] = value
                return deserialized

            if not batch:
                return []

            # Handle RAW_ITEMS format - items stored in "item" column
            if fmt == BatchFormat.RAW_ITEMS or "item" in batch:
                items = batch.get("item", [])
                return list(items) if items is not None else []

            # Handle DICT_OF_LISTS -> LIST_OF_DICTS conversion
            keys = list(batch.keys())
            if not keys:
                return []

            # Get length from first column
            first_col = batch[keys[0]]
            if first_col is None:
                return []

            n_items = len(first_col)
            items = [{k: batch[k][i] for k in keys} for i in range(n_items)]

            # Deserialize any pickled values (e.g., ObjectRefs)
            return [deserialize_row(item) for item in items]

        def _items_to_batch(self, items: List[Any], fmt: str) -> Dict[str, Any]:
            """Convert list of items to Ray Data batch (dict of columns)."""
            import pickle
            import ray

            def serialize_value(value):
                """Serialize ObjectRefs for PyArrow transport between stages.

                PyArrow doesn't natively handle ObjectRefs, and Ray Data's
                fallback pickling doesn't always work reliably.
                """
                if isinstance(value, ray._raylet.ObjectRef):
                    return b"__PICKLED_OBJREF__" + pickle.dumps(value)
                return value

            def serialize_item(item):
                if isinstance(item, dict):
                    return {k: serialize_value(v) for k, v in item.items()}
                return item

            if not items:
                return {}

            # Handle RAW_ITEMS format
            if fmt == BatchFormat.RAW_ITEMS:
                return {"item": [serialize_item(i) for i in items]}

            # Handle LIST_OF_DICTS -> DICT_OF_LISTS conversion
            if isinstance(items[0], dict):
                serialized_items = [serialize_item(item) for item in items]
                keys = serialized_items[0].keys()
                return {k: [item[k] for item in serialized_items] for k in keys}

            # Fallback: wrap as raw items
            return {"item": [serialize_item(i) for i in items]}

        def __del__(self):
            if hasattr(self, "agent"):
                self.agent.teardown()

    return AgentCallable
