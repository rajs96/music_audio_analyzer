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
"""

import inspect
from typing import List, Optional, TYPE_CHECKING

import ray.data as rd
from loguru import logger

from .agent import Agent, AgentStage, create_agent_callable

if TYPE_CHECKING:
    from .streaming_datasource import StreamingDatasource


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

        # Build compute strategy
        compute = rd.ActorPoolStrategy(
            size=stage.config.num_actors,
            max_tasks_in_flight_per_actor=stage.config.max_concurrency,
        )

        # Apply transformation
        return ds.map_batches(
            callable_class,
            batch_size=stage.config.batch_size,
            compute=compute,
            num_cpus=stage.config.num_cpus,
            num_gpus=stage.config.num_gpus,
        )

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

        # Only extract attributes that match __init__ parameters
        for key, value in agent.__dict__.items():
            if key in init_params:
                kwargs[key] = value

        return kwargs

    def stream(self, batch_size: int = 1) -> "StreamingIterator":
        """
        Start streaming through the pipeline.

        Returns an iterator that yields batches of results.
        """
        if self._dataset is None:
            self.build()

        return StreamingIterator(self._dataset, batch_size)

    def stop(self):
        """Stop the pipeline and clean up resources."""
        if self.datasource:
            self.datasource.stop()
        logger.info(f"Pipeline {self.name} stopped")


class StreamingIterator:
    """
    Iterator for streaming results from a pipeline.

    Wraps Ray Data's iterator to provide a clean interface.
    """

    def __init__(self, dataset: rd.Dataset, batch_size: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self._iter = None

    def __iter__(self):
        self._iter = self.dataset.iter_batches(
            batch_size=self.batch_size,
            prefetch_batches=2,
        )
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = self.dataset.iter_batches(
                batch_size=self.batch_size,
                prefetch_batches=2,
            )
        return next(self._iter)
