"""
Streaming Pipeline Framework

A framework for building streaming data pipelines with Ray Data.

This package provides:
    - Agent: Base class for processing logic (the recommended pattern)
    - StreamingDatasource: Base class for continuous data sources
    - StreamingPipeline: Orchestrates datasources and agents

Key concepts:
    1. Agents define pure processing logic (setup, process_batch)
    2. Datasources yield blocks continuously (infinite generators)
    3. Pipelines compose datasources and agents automatically

Example:
    from streaming_pipeline import (
        Agent,
        AgentRayComputeConfig,
        AgentStage,
        QueueStreamingDatasource,
        StreamingDatasourceConfig,
        StreamingPipeline,
    )

    # Define an agent with just the processing logic
    class MyProcessor(Agent):
        def __init__(self, multiplier: int = 2):
            super().__init__()
            self.multiplier = multiplier

        def process_batch(self, items):
            return [x * self.multiplier for x in items]

    # Create a datasource from a Ray Queue
    datasource = QueueStreamingDatasource(
        queue=my_queue,
        item_to_row_fn=lambda item: {"value": item},
        config=StreamingDatasourceConfig(batch_size=32)
    )

    # Build the pipeline - framework creates components automatically
    pipeline = StreamingPipeline(
        datasource=datasource,
        stages=[
            AgentStage(MyProcessor(multiplier=3), AgentRayComputeConfig(num_actors=4)),
        ]
    )

    # Stream results
    for batch in pipeline.stream():
        handle_results(batch)
"""

from .streaming_datasource import (
    STOP_SENTINEL,
    StreamingDatasource,
    StreamingDatasourceConfig,
    QueueStreamingDatasource,
    _deserialize_row_from_pyarrow,
    _serialize_row_for_pyarrow,
)

from .agent import (
    Agent,
    AgentRayComputeConfig,
    AgentStage,
    BatchFormat,
    FunctionAgent,
    create_agent_callable,
)

from .streaming_component import (
    StreamingIterator,
    StreamingPipeline,
)

__all__ = [
    # Datasource
    "STOP_SENTINEL",
    "StreamingDatasource",
    "StreamingDatasourceConfig",
    "QueueStreamingDatasource",
    # Agent
    "Agent",
    "AgentRayComputeConfig",
    "AgentStage",
    "BatchFormat",
    "FunctionAgent",
    "create_agent_callable",
    # Pipeline
    "StreamingIterator",
    "StreamingPipeline",
]
