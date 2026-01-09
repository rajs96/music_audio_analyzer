"""
Base class for vLLM-based audio processing agents.

Provides common functionality for:
- vLLM initialization and teardown
- Waveform extraction from pipeline items
- Batch processing flow
- Timing metrics
- Error handling

Subclasses implement:
- DETECTOR_CLS: The model/detector class to use
- get_default_sampling_kwargs(): Default vLLM sampling parameters
- parse_output(prediction): Parse model output to structured data
- get_success_fields(parsed): Extract fields for success result
- get_error_fields(): Default fields for error result
"""

import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from loguru import logger

from src.streaming_pipeline import Agent


class BaseVLLMAudioAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Base agent for vLLM-based audio processing.

    Handles all the boilerplate for vLLM inference:
    - Model loading via DETECTOR_CLS
    - Waveform extraction from pipeline items
    - Batch processing and timing
    - Error handling for PyArrow serialization quirks

    Subclasses must define:
        DETECTOR_CLS: Class that handles model loading and generation
        get_default_sampling_kwargs(): Returns default vLLM sampling params
        parse_output(prediction): Parses model output string to dict
        get_success_fields(parsed): Returns dict of fields for success result
        get_error_fields(): Returns dict of default fields for error result
    """

    DETECTOR_CLS: Type = None

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        dtype: torch.dtype = torch.bfloat16,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        # vLLM options
        use_vllm: bool = False,
        tensor_parallel_size: Optional[int] = None,
        pipeline_parallel_size: int = 1,
        distributed_executor_backend: str = "mp",
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 32768,
        max_num_seqs: int = 8,
    ):
        super().__init__()
        self.model_name = model_name
        self.dtype = dtype
        self.use_vllm = use_vllm
        self.sampling_kwargs = sampling_kwargs or self.get_default_sampling_kwargs()
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.distributed_executor_backend = distributed_executor_backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.detector = None
        self.total_inference_time_ms = 0.0
        self.batch_count = 0
        self.processed_count = 0

    @abstractmethod
    def get_default_sampling_kwargs(self) -> Dict[str, Any]:
        """Return default vLLM sampling parameters."""
        pass

    @abstractmethod
    def parse_output(self, prediction: str) -> Dict[str, Any]:
        """Parse model output string to structured data."""
        pass

    @abstractmethod
    def get_success_fields(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return task-specific fields for a successful result.

        These fields are merged with the common fields (job_id, song_id, etc.)
        """
        pass

    @abstractmethod
    def get_error_fields(self) -> Dict[str, Any]:
        """
        Return default task-specific fields for an error result.

        These fields are merged with the common fields (job_id, song_id, etc.)
        """
        pass

    def setup(self) -> None:
        """Load the detector - called once per actor."""
        if self.DETECTOR_CLS is None:
            raise ValueError(f"{self.__class__.__name__} must define DETECTOR_CLS")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        backend = "vLLM" if self.use_vllm else "HuggingFace"
        logger.info(
            f"{self.__class__.__name__} loading detector {self.model_name} "
            f"on {device} with dtype={self.dtype}, backend={backend}"
        )

        if self.use_vllm:
            logger.info(
                f"vLLM config: tp={self.tensor_parallel_size}, "
                f"pp={self.pipeline_parallel_size}, "
                f"backend={self.distributed_executor_backend}"
            )
            self.detector = self.DETECTOR_CLS(
                model_name=self.model_name,
                dtype=self.dtype,
                use_vllm=True,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                distributed_executor_backend=self.distributed_executor_backend,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
            )
        else:
            self.detector = self.DETECTOR_CLS(
                model_name=self.model_name,
                dtype=self.dtype,
                device=device,
            )
        logger.info(f"{self.__class__.__name__} detector loaded ({backend})")

    def teardown(self) -> None:
        """Clean up detector resources."""
        if self.detector is not None:
            self.detector.unload()
            self.detector = None
            logger.info(f"{self.__class__.__name__} cleaned up")

    def _get_waveforms_from_items(
        self, items: List[Dict[str, Any]], now: int
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[np.ndarray]]:
        """
        Extract waveforms from items, separating valid from errored.

        Waveforms are passed as serialized bytes (waveform_bytes) and deserialized
        using np.frombuffer(). No ray.get() calls needed.

        Returns:
            (error_results, valid_items, valid_waveforms)
        """
        results = []
        valid_items = []
        valid_waveforms = []

        for idx, item in enumerate(items):
            filename = item.get("filename", f"unknown_{idx}")

            # PyArrow converts None to nan during serialization
            error_value = item.get("error")
            has_real_error = (
                error_value is not None
                and error_value != "nan"
                and not (isinstance(error_value, float) and str(error_value) == "nan")
                and str(error_value).strip() != ""
            )
            if has_real_error:
                logger.warning(
                    f"Item '{filename}' has preprocessing error: {error_value}"
                )
                results.append(self._create_error_result(item, now, str(error_value)))
                continue

            try:
                waveform_bytes = item.get("waveform_bytes")

                if waveform_bytes is None or len(waveform_bytes) == 0:
                    raise ValueError("waveform_bytes is None or empty")

                # Handle PyArrow binary types - convert to Python bytes if needed
                if hasattr(waveform_bytes, "as_py"):
                    waveform_bytes = waveform_bytes.as_py()
                elif hasattr(waveform_bytes, "tobytes"):
                    waveform_bytes = waveform_bytes.tobytes()

                waveform = np.frombuffer(waveform_bytes, dtype=np.float32)

                if waveform.size == 0:
                    raise ValueError("Waveform is empty (size=0)")

                logger.debug(
                    f"Item '{filename}' waveform deserialized: shape={waveform.shape}, "
                    f"dtype={waveform.dtype}"
                )

                valid_items.append(item)
                valid_waveforms.append(waveform)

            except Exception as e:
                logger.error(f"Failed to deserialize waveform for '{filename}': {e}")
                results.append(
                    self._create_error_result(
                        item, now, f"Failed to deserialize waveform: {str(e)}"
                    )
                )

        return results, valid_items, valid_waveforms

    def _create_error_result(
        self, item: Dict[str, Any], now: int, error: str
    ) -> Dict[str, Any]:
        """Create an error result for an item."""
        result = {
            "job_id": item["job_id"],
            "song_id": item["song_id"],
            "song_hash": item["song_hash"],
            "filename": item["filename"],
            "detected_at": now,
            "error": error,
        }
        result.update(self.get_error_fields())
        return result

    def _create_success_result(
        self, item: Dict[str, Any], now: int, parsed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a success result for an item."""
        result = {
            "job_id": item["job_id"],
            "song_id": item["song_id"],
            "song_hash": item["song_hash"],
            "filename": item["filename"],
            "detected_at": now,
            "error": "",
        }
        result.update(self.get_success_fields(parsed))
        return result

    def predict_batch_internal(self, waveforms: List[np.ndarray]) -> List[str]:
        """Run inference on a batch of waveforms using detector."""
        if self.use_vllm:
            responses = self.detector.generate(
                waveforms=waveforms,
                sampling_kwargs=self.sampling_kwargs,
            )
        else:
            inputs = self._tokenize_hf(waveforms)
            processed_inputs = self._process_hf_inputs(inputs)

            with torch.no_grad():
                responses = self.detector.generate(
                    inputs=processed_inputs,
                    generate_kwargs=self.sampling_kwargs,
                )

        return responses

    def _tokenize_hf(self, waveforms: List[np.ndarray]) -> Dict[str, Any]:
        """Convert waveforms to HF model inputs."""
        conversations = [
            self.detector.DATASET_CLS.get_conversation(waveform)
            for waveform in waveforms
        ]

        inputs = self.detector.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return inputs

    def _process_hf_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Move HF inputs to device with correct dtype."""
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    processed_inputs[k] = v.to(
                        device=self.detector.device, dtype=self.dtype
                    )
                else:
                    processed_inputs[k] = v.to(device=self.detector.device)
            else:
                processed_inputs[k] = v
        return processed_inputs

    def process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of preprocessed audio through the detector."""
        if not items:
            return []

        now = int(time.time())
        results, valid_items, valid_waveforms = self._get_waveforms_from_items(
            items, now
        )

        if valid_waveforms:
            inference_start = time.time()
            predictions = self.predict_batch_internal(valid_waveforms)
            inference_time_ms = (time.time() - inference_start) * 1000
            self.total_inference_time_ms += inference_time_ms
            self.batch_count += 1
            self.processed_count += len(valid_items)

            logger.info(
                f"Batch inference: {inference_time_ms:.1f}ms "
                f"({inference_time_ms/len(valid_items):.1f}ms/example)"
            )

            for item, prediction in zip(valid_items, predictions):
                parsed = self.parse_output(prediction)
                results.append(self._create_success_result(item, now, parsed))

                logger.debug(f"Processed {item['filename']}: {parsed}")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get timing statistics."""
        avg_batch_time_ms = (
            self.total_inference_time_ms / self.batch_count
            if self.batch_count > 0
            else 0.0
        )
        avg_per_example_ms = (
            self.total_inference_time_ms / self.processed_count
            if self.processed_count > 0
            else 0.0
        )
        return {
            "processed_count": self.processed_count,
            "batch_count": self.batch_count,
            "total_inference_time_ms": self.total_inference_time_ms,
            "avg_batch_time_ms": avg_batch_time_ms,
            "avg_per_example_ms": avg_per_example_ms,
        }
