"""
Instrument Detector Agent.

Runs ML inference to detect instruments in preprocessed audio.
Inherits from BaseVLLMAudioAgent for common vLLM/batch processing logic.
"""

import ast
import json
import time
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from loguru import logger

from src.base_agents import BaseVLLMAudioAgent
from src.models.qwen_instrument_detector import (
    QwenOmniInstrumentDetector,
    QwenOmniCoTInstrumentDetector,
)


DEFAULT_VLLM_SAMPLING_KWARGS = {
    "temperature": 0.0,
    "max_tokens": 256,
}

DEFAULT_VLLM_COT_PLANNING_KWARGS = {
    "temperature": 0.0,
    "max_tokens": 256,
}

DEFAULT_VLLM_COT_RESPONSE_KWARGS = {
    "temperature": 0.0,
    "max_tokens": 128,
}


class InstrumentDetectorAgent(BaseVLLMAudioAgent):
    """
    Agent that runs instrument detection on preprocessed audio.

    Uses QwenOmniInstrumentDetector for model loading and inference.
    Inherits vLLM/batch processing logic from BaseVLLMAudioAgent.

    Input format:
        {
            "job_id": str,
            "song_id": str,
            "song_hash": str,
            "filename": str,
            "waveform_bytes": bytes,
        }

    Output format:
        {
            "job_id": str,
            "song_id": str,
            "song_hash": str,
            "filename": str,
            "instruments": List[str],
            "detected_at": int,
        }
    """

    DETECTOR_CLS: Type[QwenOmniInstrumentDetector] = QwenOmniInstrumentDetector

    def get_default_sampling_kwargs(self) -> Dict[str, Any]:
        """Return default vLLM sampling parameters for instrument detection."""
        return DEFAULT_VLLM_SAMPLING_KWARGS.copy()

    def parse_output(self, prediction: str) -> Dict[str, Any]:
        """Parse the model output to extract instrument list."""
        try:
            start = prediction.find("[")
            end = prediction.rfind("]") + 1
            if start != -1 and end > start:
                instruments = ast.literal_eval(prediction[start:end])
                return {"instruments": instruments}
        except Exception:
            pass
        return {"instruments": [prediction.strip()]}

    def get_success_fields(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Return instrument detection specific fields for success result."""
        return {
            "instruments": parsed.get("instruments", []),
        }

    def get_error_fields(self) -> Dict[str, Any]:
        """Return default instrument detection fields for error result."""
        return {
            "instruments": [],
        }


class InstrumentDetectorCoTAgent(BaseVLLMAudioAgent):
    """
    Agent that runs chain-of-thought instrument detection on preprocessed audio.

    Uses QwenOmniCoTInstrumentDetector for two-step reasoning:
    1. Describe sounds in background/middle-ground/foreground layers
    2. Convert descriptions to structured JSON with instruments

    This agent overrides process_batch because it has two-step inference
    (planning + response) which differs from the single-step base class.

    Output format includes layer-based instruments:
        {
            "job_id": str,
            "song_id": str,
            "song_hash": str,
            "filename": str,
            "background": List[str],
            "middle_ground": List[str],
            "foreground": List[str],
            "instruments": List[str],  # flattened for backwards compat
            "planning_response": str,  # step 1 response
            "detected_at": int,
        }
    """

    DETECTOR_CLS: Type[QwenOmniCoTInstrumentDetector] = QwenOmniCoTInstrumentDetector

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        dtype: torch.dtype = torch.bfloat16,
        # vLLM options
        use_vllm: bool = False,
        planning_sampling_kwargs: Optional[Dict[str, Any]] = None,
        response_sampling_kwargs: Optional[Dict[str, Any]] = None,
        tensor_parallel_size: Optional[int] = None,
        pipeline_parallel_size: int = 1,
        distributed_executor_backend: str = "mp",
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 32768,
        max_num_seqs: int = 8,
    ):
        super().__init__(
            model_name=model_name,
            dtype=dtype,
            use_vllm=use_vllm,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )

        self.planning_sampling_kwargs = (
            planning_sampling_kwargs or DEFAULT_VLLM_COT_PLANNING_KWARGS.copy()
        )
        self.response_sampling_kwargs = (
            response_sampling_kwargs or DEFAULT_VLLM_COT_RESPONSE_KWARGS.copy()
        )

    def get_default_sampling_kwargs(self) -> Dict[str, Any]:
        return DEFAULT_VLLM_COT_PLANNING_KWARGS.copy()

    def parse_output(self, prediction: str) -> Dict[str, Any]:
        """Parse the model output to extract layered instrument JSON."""
        logger.debug(f"Parsing CoT prediction: {prediction[:500]}...")

        try:
            start = prediction.find("{")
            end = prediction.rfind("}") + 1
            if start != -1 and end > start:
                json_str = prediction[start:end]
                logger.debug(f"Extracted JSON: {json_str}")
                parsed = json.loads(json_str)

                # Extract layers
                background = parsed.get("background", [])
                middle_ground = parsed.get("middle_ground", [])
                foreground = parsed.get("foreground", [])

                # Flatten for backwards compat
                all_instruments = list(
                    set(background) | set(middle_ground) | set(foreground)
                )

                logger.info(
                    f"Parsed instruments - bg: {background}, mid: {middle_ground}, "
                    f"fg: {foreground}, all: {all_instruments}"
                )

                return {
                    "background": background,
                    "middle_ground": middle_ground,
                    "foreground": foreground,
                    "instruments": all_instruments,
                }
            else:
                logger.warning(
                    f"No JSON object found in prediction. "
                    f"Raw prediction: {prediction[:200]}..."
                )
        except Exception as e:
            logger.warning(
                f"Failed to parse CoT response: {e}. "
                f"Raw prediction: {prediction[:200]}..."
            )

        return {
            "background": [],
            "middle_ground": [],
            "foreground": [],
            "instruments": [],
        }

    def get_success_fields(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Return CoT-specific fields for success result."""
        return {
            "background": parsed.get("background", []),
            "middle_ground": parsed.get("middle_ground", []),
            "foreground": parsed.get("foreground", []),
            "instruments": parsed.get("instruments", []),
            "planning_response": parsed.get("planning_response", ""),
        }

    def get_error_fields(self) -> Dict[str, Any]:
        """Return default CoT fields for error result."""
        return {
            "background": [],
            "middle_ground": [],
            "foreground": [],
            "instruments": [],
            "planning_response": "",
        }

    def predict_batch_internal(
        self, waveforms: List[np.ndarray]
    ) -> tuple[List[str], List[str]]:
        planning_responses, final_responses = self.detector.generate(
            waveforms=waveforms,
            planning_sampling_kwargs=self.planning_sampling_kwargs,
            response_sampling_kwargs=self.response_sampling_kwargs,
        )
        return planning_responses, final_responses

    def process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"InstrumentDetectorCoT received batch of {len(items)} items")
        if not items:
            return []

        now = int(time.time())
        results, valid_items, valid_waveforms = self._get_waveforms_from_items(
            items, now
        )
        logger.info(f"Extracted {len(valid_waveforms)} valid waveforms from batch")

        if valid_waveforms:
            logger.info(
                f"Starting vLLM inference on {len(valid_waveforms)} waveforms..."
            )
            inference_start = time.time()
            planning_responses, final_responses = self.predict_batch_internal(
                valid_waveforms
            )
            inference_time_ms = (time.time() - inference_start) * 1000
            self.total_inference_time_ms += inference_time_ms
            self.batch_count += 1
            self.processed_count += len(valid_items)

            logger.info(
                f"Batch inference: {inference_time_ms:.1f}ms "
                f"({inference_time_ms/len(valid_items):.1f}ms/example)"
            )

            for item, planning, final in zip(
                valid_items, planning_responses, final_responses
            ):
                logger.info(
                    f"Planning response for {item['filename']}: {planning[:200]}..."
                )
                logger.info(f"Final response for {item['filename']}: {final[:200]}...")

                parsed = self.parse_output(final)
                parsed["planning_response"] = planning
                results.append(self._create_success_result(item, now, parsed))

                logger.info(f"Detected instruments for {item['filename']}: {parsed}")

        return results
