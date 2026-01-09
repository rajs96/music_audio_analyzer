# Agent Class Structure

This document describes the agentic design for audio processing pipelines.

## Architecture Overview

```
src/
├── base_agents/                    # Reusable base classes
│   ├── __init__.py
│   ├── audio_preprocessor.py       # AudioPreprocessorAgent
│   └── base_vllm_audio_agent.py    # BaseVLLMAudioAgent
│
├── streaming_pipeline/             # Ray Data pipeline infrastructure
│   ├── agent.py                    # Agent base class, AgentStage
│   ├── streaming_datasource.py     # Queue-based data source
│   └── streaming_pipeline.py       # Pipeline orchestration
│
└── pipelines/
    └── instrument_detection/
        └── agents/
            ├── audio_preprocessor.py   # Re-exports from base_agents
            └── instrument_detector.py  # InstrumentDetectorAgent, InstrumentDetectorCoTAgent
```

## Class Hierarchy

```
Agent (ABC)
│
├── AudioPreprocessorAgent
│   └── Decodes audio bytes → waveform bytes
│
└── BaseVLLMAudioAgent (ABC)
    │   Common vLLM inference logic
    │
    ├── InstrumentDetectorAgent
    │   └── Single-step instrument detection
    │
    └── InstrumentDetectorCoTAgent
        └── Two-step chain-of-thought detection
```

## Base Classes

### Agent (streaming_pipeline/agent.py)

The foundation for all pipeline stages.

```python
class Agent(ABC, Generic[TInput, TOutput]):
    def setup(self) -> None: ...
    def teardown(self) -> None: ...
    def process(self, item: TInput) -> TOutput: ...
    @abstractmethod
    def process_batch(self, items: List[TInput]) -> List[TOutput]: ...
```

### AudioPreprocessorAgent (base_agents/audio_preprocessor.py)

Decodes audio bytes to waveforms. Stateless, CPU-only.

```python
class AudioPreprocessorAgent(Agent):
    def __init__(self, target_sr: int = 16000): ...
    def decode_audio_bytes_to_waveform(self, audio_bytes, suffix) -> np.ndarray: ...
    def process_batch(self, items) -> List[Dict]: ...
```

**Input:**
```python
{
    "job_id": str,
    "song_id": str,
    "song_hash": str,
    "filename": str,
    "audio_bytes": bytes,
}
```

**Output:**
```python
{
    "job_id": str,
    "song_id": str,
    "song_hash": str,
    "filename": str,
    "waveform_bytes": bytes,  # np.frombuffer(bytes, dtype=np.float32)
    "sample_rate": int,
    "duration_seconds": float,
    "error": str,
}
```

### BaseVLLMAudioAgent (base_agents/base_vllm_audio_agent.py)

Abstract base for vLLM-based audio inference agents.

```python
class BaseVLLMAudioAgent(Agent, ABC):
    DETECTOR_CLS: Type = None  # Subclasses set this

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype,
        use_vllm: bool,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_seqs: int,
        ...
    ): ...

    # Subclasses implement these:
    @abstractmethod
    def get_default_sampling_kwargs(self) -> Dict: ...
    @abstractmethod
    def parse_output(self, prediction: str) -> Dict: ...
    @abstractmethod
    def get_success_fields(self, parsed: Dict) -> Dict: ...
    @abstractmethod
    def get_error_fields(self) -> Dict: ...

    # Provided by base class:
    def setup(self) -> None: ...           # Loads model
    def teardown(self) -> None: ...        # Unloads model
    def process_batch(self, items) -> List[Dict]: ...
    def predict_batch_internal(self, waveforms) -> List[str]: ...
    def get_stats(self) -> Dict: ...
```

## Concrete Agents

### InstrumentDetectorAgent

Single-step instrument detection.

```python
class InstrumentDetectorAgent(BaseVLLMAudioAgent):
    DETECTOR_CLS = QwenOmniInstrumentDetector

    def get_default_sampling_kwargs(self):
        return {"temperature": 0.0, "max_tokens": 256}

    def parse_output(self, prediction: str) -> Dict:
        # Extracts ["guitar", "drums", ...] from model output
        ...

    def get_success_fields(self, parsed: Dict) -> Dict:
        return {"instruments": parsed.get("instruments", [])}

    def get_error_fields(self) -> Dict:
        return {"instruments": []}
```

### InstrumentDetectorCoTAgent

Two-step chain-of-thought detection with layered output.

```python
class InstrumentDetectorCoTAgent(BaseVLLMAudioAgent):
    DETECTOR_CLS = QwenOmniCoTInstrumentDetector

    def __init__(self, ..., planning_sampling_kwargs, response_sampling_kwargs): ...

    def predict_batch_internal(self, waveforms) -> tuple[List[str], List[str]]:
        # Returns (planning_responses, final_responses)
        ...

    def parse_output(self, prediction: str) -> Dict:
        # Extracts {background: [...], middle_ground: [...], foreground: [...]}
        ...

    def get_success_fields(self, parsed: Dict) -> Dict:
        return {
            "background": parsed.get("background", []),
            "middle_ground": parsed.get("middle_ground", []),
            "foreground": parsed.get("foreground", []),
            "instruments": parsed.get("instruments", []),
            "planning_response": parsed.get("planning_response", ""),
        }

    def process_batch(self, items) -> List[Dict]:
        # Custom override for two-step inference
        ...
```

## Creating a New Agent

To create a new vLLM-based audio agent (e.g., MoodAnalyzerAgent):

```python
from src.base_agents import BaseVLLMAudioAgent
from src.models.qwen_mood_analyzer import QwenOmniMoodAnalyzer

class MoodAnalyzerAgent(BaseVLLMAudioAgent):
    DETECTOR_CLS = QwenOmniMoodAnalyzer

    def get_default_sampling_kwargs(self):
        return {"temperature": 0.0, "max_tokens": 64}

    def parse_output(self, prediction: str) -> dict:
        # Parse mood/energy from model output
        return {"mood": prediction.strip(), "energy": 0.5}

    def get_success_fields(self, parsed: dict) -> dict:
        return {"mood": parsed["mood"], "energy": parsed["energy"]}

    def get_error_fields(self) -> dict:
        return {"mood": "", "energy": 0.0}
```

The base class handles:
- Model loading/unloading
- Waveform extraction from pipeline items
- Batch processing and timing
- Error handling (PyArrow nan issues)
- Stats collection

## Pipeline Assembly

```python
from src.streaming_pipeline import (
    AgentStage,
    AgentRayComputeConfig,
    StreamingPipeline,
    QueueStreamingDatasource,
)

# Create stages
preprocessor = AgentStage(
    agent=AudioPreprocessorAgent(target_sr=16000),
    config=AgentRayComputeConfig(num_actors=4, batch_size=8, num_cpus=1.0),
)

detector = AgentStage(
    agent=InstrumentDetectorCoTAgent(
        model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        use_vllm=True,
        tensor_parallel_size=1,
    ),
    config=AgentRayComputeConfig(num_actors=1, batch_size=4, num_gpus=1.0),
)

# Assemble pipeline
pipeline = StreamingPipeline(
    datasource=QueueStreamingDatasource(queue, ...),
    stages=[preprocessor, detector],
)

# Run
for batch in pipeline.stream():
    process_results(batch)
```

## Data Flow

```
┌─────────────────┐
│   Job Queue     │
│  (audio_bytes)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessor    │  CPU actors (4x)
│ decode → resample│
└────────┬────────┘
         │ waveform_bytes
         ▼
┌─────────────────┐
│ Detector        │  GPU actor (1x)
│ vLLM inference  │
└────────┬────────┘
         │ instruments
         ▼
┌─────────────────┐
│ Results         │
│ background: []  │
│ middle_ground: []│
│ foreground: []  │
└─────────────────┘
```

## Key Design Decisions

1. **Waveform as bytes**: Waveforms are serialized as `bytes` using `tobytes()` and deserialized with `np.frombuffer()`. This avoids Ray ObjectRef ownership issues when actors shut down.

2. **Empty string for no error**: Use `error: ""` instead of `error: None` to avoid PyArrow converting None to nan during serialization.

3. **Abstract base for vLLM agents**: Common model loading, batch processing, and timing logic lives in `BaseVLLMAudioAgent`. Subclasses only implement parsing and field extraction.

4. **Two-step CoT override**: `InstrumentDetectorCoTAgent` overrides `process_batch` because its two-step inference pattern differs from the single-step base class flow.
