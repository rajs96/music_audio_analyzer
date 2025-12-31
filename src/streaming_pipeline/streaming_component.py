"""
StreamingComponent and Pipeline orchestration.

This module provides:
    - StreamingPipeline: Orchestrates datasources and agents
    - AgentStage integration: Wraps Agents into Ray Data transformations
    - Legacy ComponentStage: For backward compatibility

The recommended pattern is to use Agent + AgentStage:
    pipeline = StreamingPipeline(
        datasource=my_datasource,
        stages=[
            AgentStage(MyAgent(), AgentRayComputeConfig(num_actors=4)),
        ]
    )

For vLLM with tensor parallelism:
    pipeline = StreamingPipeline(
        datasource=my_datasource,
        stages=[
            AgentStage(
                MyVLLMAgent(),
                AgentRayComputeConfig(
                    num_actors=1,
                    tensor_parallel_size=4,
                    batch_size=8,
                )
            ),
        ]
    )
"""

import inspect
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Iterator

import ray
import ray.data as rd
from loguru import logger
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .agent import Agent, AgentStage, AgentRayComputeConfig, create_agent_callable

if TYPE_CHECKING:
    from .streaming_datasource import StreamingDatasource


def _create_placement_group_strategy(
    num_bundles_per_replica: int,
    accelerator_type: Optional[str] = None,
    strategy: str = "PACK",
) -> Dict[str, Any]:
    """
    Create a Ray scheduling strategy with placement groups for tensor/pipeline parallelism.

    This is used when distributed_executor_backend="ray" to let vLLM's ray executor
    manage GPU allocation across nodes.

    Args:
        num_bundles_per_replica: Number of GPU bundles (tp_size * pp_size)
        accelerator_type: Optional GPU type label (e.g., "A100", "H100")
        strategy: Placement group strategy ("PACK", "STRICT_PACK", "SPREAD", "STRICT_SPREAD")

    Returns:
        Dict with scheduling_strategy for Ray remote args
    """
    # Create bundle specification
    bundle = {"GPU": 1, "CPU": 1}
    if accelerator_type:
        bundle[f"accelerator_type:{accelerator_type}"] = 0.001

    # Create placement group with specified number of bundles
    pg = ray.util.placement_group(
        [bundle] * num_bundles_per_replica,
        strategy=strategy,
    )

    return {
        "scheduling_strategy": PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True
        )
    }


class StreamingPipeline:
    """
    A pipeline composed of Agents.

    The pipeline connects a StreamingDatasource to a series of
    processing stages (Agents), creating an end-to-end streaming system.

    Example:
        from streaming_pipeline import (
            StreamingPipeline,
            QueueStreamingDatasource,
            AgentStage,
            AgentRayComputeConfig,
        )
        from my_agents import AudioDecoder, InstrumentDetector

        # Create datasource
        datasource = QueueStreamingDatasource(
            queue=job_queue,
            item_to_row_fn=lambda job: {"job_id": job.id, ...}
        )

        # Create agent stages
        stages = [
            AgentStage(AudioDecoder(), AgentRayComputeConfig(num_actors=4)),
            AgentStage(InstrumentDetector(), AgentRayComputeConfig(num_actors=1, num_gpus=1)),
        ]

        # Create and run pipeline
        pipeline = StreamingPipeline(datasource, stages)
        for result in pipeline.stream():
            handle_result(result)
    """

    def __init__(
        self,
        datasource: "StreamingDatasource",
        stages: List[AgentStage],
        name: str = "StreamingPipeline",
    ):
        self.datasource = datasource
        self.stages = stages
        self.name = name
        self._dataset: Optional[rd.Dataset] = None

    def build(self) -> rd.Dataset:
        """
        Build the Ray Dataset pipeline.

        Returns the final dataset with all transformations applied.
        The dataset is lazy - no processing happens until iteration.
        """
        # Configure Ray Data for true streaming execution
        # This prevents buffering all blocks before processing starts
        ctx = rd.DataContext.get_current()
        # Note: In Ray 2.5+, streaming execution is the default (use_streaming_executor removed)
        ctx.execution_options.preserve_order = (
            False  # Allow out-of-order for throughput
        )
        ctx.target_min_block_size = 1  # Don't wait for large blocks
        ctx.target_max_block_size = 1024 * 1024  # 1MB max block size
        logger.info(
            f"Ray Data streaming config: preserve_order={ctx.execution_options.preserve_order}"
        )

        # Create initial dataset from datasource
        ds = rd.read_datasource(self.datasource)

        # Apply each stage
        for stage in self.stages:
            logger.info(f"Adding stage: {stage.name}")
            ds = self._apply_agent_stage(ds, stage)

        self._dataset = ds
        return ds

    def _apply_agent_stage(self, ds: rd.Dataset, stage: AgentStage) -> rd.Dataset:
        """
        Apply an AgentStage as a transformation on a Ray Dataset.

        This creates the callable wrapper and configures Ray Data's
        map_batches with the appropriate resources.

        For vLLM with tensor/pipeline parallelism:
        - Uses placement groups when distributed_executor_backend="ray"
        - Sets num_gpus = tensor_parallel_size * pipeline_parallel_size
        """
        # Get agent class and init kwargs
        agent = stage.agent
        agent_class = agent.__class__
        agent_kwargs = self._extract_init_kwargs(agent)

        # Create the callable class for map_batches
        callable_class = create_agent_callable(
            agent_class=agent_class,
            agent_args=(),
            agent_kwargs=agent_kwargs,
            input_format=stage.input_format,
            output_format=stage.output_format,
        )

        config = stage.config

        # Build compute strategy
        compute = rd.ActorPoolStrategy(
            size=config.num_actors,
            max_tasks_in_flight_per_actor=config.max_concurrency,
        )

        # Build map_batches kwargs
        map_kwargs: Dict[str, Any] = {
            "batch_size": config.batch_size,
            "compute": compute,
            "num_cpus": config.num_cpus,
        }

        # Handle GPU allocation based on parallelism settings
        num_bundles = config.get_num_bundles_per_replica()

        if config.distributed_executor_backend == "ray" and num_bundles > 1:
            # Use placement groups for ray distributed backend
            # Ray Data won't reserve GPUs - vLLM's ray executor creates placement groups
            map_kwargs["num_gpus"] = 0
            map_kwargs["ray_remote_args_fn"] = partial(
                _create_placement_group_strategy,
                num_bundles_per_replica=num_bundles,
                accelerator_type=config.accelerator_type,
                strategy=config.placement_group_strategy,
            )
            logger.info(
                f"Stage {stage.name}: Using ray backend with placement groups "
                f"({num_bundles} GPU bundles, strategy={config.placement_group_strategy})"
            )
        else:
            # Default "mp" backend: set num_gpus directly
            num_gpus = config.get_num_gpus_per_actor()
            map_kwargs["num_gpus"] = num_gpus
            if num_bundles > 1:
                logger.info(
                    f"Stage {stage.name}: Using mp backend with {num_gpus} GPUs per actor "
                    f"(tp={config.tensor_parallel_size}, pp={config.pipeline_parallel_size})"
                )

        # Add accelerator type if specified
        if config.accelerator_type:
            map_kwargs["accelerator_type"] = config.accelerator_type

        # Apply transformation
        return ds.map_batches(callable_class, **map_kwargs)

    def _extract_init_kwargs(self, agent: Agent) -> dict:
        """
        Extract initialization kwargs from an agent instance.

        This inspects the agent's __init__ signature to find only the
        parameters that should be passed when recreating the agent.
        Only includes attributes that match __init__ parameter names.
        """
        kwargs = {}

        # Get the __init__ signature of the agent class
        agent_class = agent.__class__
        try:
            sig = inspect.signature(agent_class.__init__)
            init_params = set(sig.parameters.keys()) - {"self"}
        except (ValueError, TypeError):
            # Fallback: if we can't inspect, use empty set
            logger.warning(
                f"Could not inspect __init__ for {agent_class.__name__}, "
                "using empty kwargs"
            )
            return {}

        # Extract attributes that match __init__ parameters
        # Check both exact match and underscore-prefixed versions
        # (e.g., process_fn stored as _process_fn)
        for param_name in init_params:
            # Check exact match first
            if param_name in agent.__dict__:
                kwargs[param_name] = agent.__dict__[param_name]
            # Check underscore-prefixed version (common convention for "private" storage)
            elif f"_{param_name}" in agent.__dict__:
                kwargs[param_name] = agent.__dict__[f"_{param_name}"]

        return kwargs

    def stream(self, batch_size: int = 1) -> "StreamingIterator":
        """
        Start streaming through the pipeline (no warmup).

        Returns an iterator that yields batches of results.
        For warmup support, use warmup_and_stream() instead.
        """
        if self._dataset is None:
            self.build()

        return StreamingIterator(self._dataset, batch_size)

    def warmup_and_stream(
        self,
        warmup_data_fn: Optional[Callable[[], List[Dict[str, Any]]]] = None,
        warmup_timeout: float = 300.0,
        batch_size: int = 1,
    ) -> Iterator[dict]:
        """
        Warm up the pipeline and then start streaming.

        IMPORTANT: This method uses a SINGLE iter_batches() call to ensure
        the same Ray Data actors handle both warmup and real data. This is
        critical for vLLM - the model loads once during warmup, then the
        same actors (with loaded models) process real data.

        The warmup items are processed first and their results are discarded.
        After warmup completes, subsequent batches are yielded to the caller.

        Args:
            warmup_data_fn: Optional function that returns warmup items.
                           These should be real, processable items.
            warmup_timeout: Maximum time for warmup in seconds.
            batch_size: Batch size for streaming results.

        Yields:
            Batches of results (after warmup items are consumed).

        Example:
            def get_warmup_data():
                return [{"job_id": "warmup", "audio_bytes": real_audio_bytes}]

            for batch in pipeline.warmup_and_stream(warmup_data_fn=get_warmup_data):
                process(batch)  # Only real results, warmup discarded
        """
        if self._dataset is None:
            self.build()

        # Determine warmup count
        num_warmup = 0
        if warmup_data_fn is not None:
            warmup_items = warmup_data_fn()
            if warmup_items:
                num_warmup = len(warmup_items)

                # Inject warmup items into the datasource queue
                if (
                    hasattr(self.datasource, "_queue")
                    and self.datasource._queue is not None
                ):
                    logger.info(
                        f"Injecting {num_warmup} warmup items into datasource queue..."
                    )
                    for item in warmup_items:
                        self.datasource._queue.put(item)
                else:
                    logger.warning(
                        "Datasource does not have accessible queue - "
                        "warmup items cannot be injected. Proceeding without warmup."
                    )
                    num_warmup = 0

        # Use generator that handles warmup + streaming with SINGLE iter_batches
        return self._warmup_streaming_generator(
            num_warmup=num_warmup,
            warmup_timeout=warmup_timeout,
            batch_size=batch_size,
        )

    def _warmup_streaming_generator(
        self,
        num_warmup: int,
        warmup_timeout: float,
        batch_size: int,
    ) -> Iterator[dict]:
        """
        Generator that handles warmup then yields real results.

        Uses a SINGLE iter_batches() call to ensure actors stay alive
        across warmup and real data processing. This is critical for
        vLLM where model loading happens on first batch.
        """
        warmup_count = 0
        warmup_start = time.time()
        warmup_complete = num_warmup == 0

        if num_warmup > 0:
            logger.info(f"Starting warmup with {num_warmup} items...")

        # SINGLE iter_batches call - actors are created ONCE and reused
        for batch in self._dataset.iter_batches(
            batch_size=batch_size,
            prefetch_batches=0,
        ):
            if not warmup_complete:
                # Still in warmup phase - discard results
                warmup_count += 1
                elapsed = time.time() - warmup_start
                logger.info(
                    f"Warmup progress: {warmup_count}/{num_warmup} ({elapsed:.1f}s)"
                )

                if elapsed > warmup_timeout:
                    logger.warning(
                        f"Warmup timed out after {elapsed:.1f}s - proceeding anyway"
                    )
                    warmup_complete = True

                if warmup_count >= num_warmup:
                    elapsed = time.time() - warmup_start
                    logger.info(f"Warmup complete in {elapsed:.1f}s - actors ready")
                    warmup_complete = True

                continue  # Discard warmup batch

            # Warmup complete - yield real results
            yield batch

    def stop(self):
        """Stop the pipeline and clean up resources."""
        if self.datasource:
            self.datasource.stop()
        logger.info(f"Pipeline {self.name} stopped")


class StreamingIterator:
    """
    Iterator for streaming results from a pipeline.

    Wraps Ray Data's iterator to provide a clean interface.
    Uses streaming-friendly settings to start processing immediately.
    """

    def __init__(self, dataset: rd.Dataset, batch_size: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self._iter = None

    def __iter__(self) -> Iterator[dict]:
        # Use prefetch_batches=0 to start processing immediately
        # instead of buffering batches before yielding
        self._iter = iter(
            self.dataset.iter_batches(
                batch_size=self.batch_size,
                prefetch_batches=0,  # Don't prefetch - yield as soon as available
            )
        )
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = iter(
                self.dataset.iter_batches(
                    batch_size=self.batch_size,
                    prefetch_batches=0,
                )
            )
        return next(self._iter)
