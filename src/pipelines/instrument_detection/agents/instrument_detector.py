"""
Instrument Detector Agent.

Runs ML inference to detect instruments in preprocessed audio.
"""

import json
import time
from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger

from src.streaming_pipeline import Agent
from src.data.qwen_omni import QwenOmniDataset, QwenOmniCoTDataset


class InstrumentDetectorAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Agent that runs instrument detection on preprocessed audio.

    The model is loaded once in setup() and reused for all batches.
    Uses QwenOmniDataset for conversation building.

    Input format:
        {
            "job_id": str,
            "song_id": str,
            "song_hash": str,
            "filename": str,
            "waveform": np.ndarray,
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

    # Dataset class for conversation building
    DATASET_CLS = QwenOmniDataset

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_name = model_name
        self.dtype = dtype

        # Will be initialized in setup()
        self.model = None
        self.processor = None
        self.device = None

        # Timing metrics
        self.total_inference_time_ms = 0.0
        self.batch_count = 0
        self.processed_count = 0

    def setup(self) -> None:
        """Load the model - called once per actor."""
        from src.models.load_qwen import load_model_and_processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"{self.__class__.__name__} loading model {self.model_name} "
            f"on {self.device} with dtype={self.dtype}"
        )

        self.model, self.processor = load_model_and_processor(
            self.model_name, self.dtype, self.device
        )
        self.model.eval()
        logger.info(f"{self.__class__.__name__} model loaded")

    def teardown(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"{self.__class__.__name__} cleaned up")

    def tokenize(self, waveforms: List[np.ndarray]) -> Dict[str, Any]:
        """Convert waveforms to model inputs using dataset conversation builder."""
        conversations = [
            self.DATASET_CLS.get_conversation(waveform) for waveform in waveforms
        ]

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return inputs

    def process_hf_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Move inputs to device with correct dtype."""
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    processed_inputs[k] = v.to(device=self.device, dtype=self.dtype)
                else:
                    processed_inputs[k] = v.to(device=self.device)
            else:
                processed_inputs[k] = v
        return processed_inputs

    def generate_responses(self, inputs: Dict[str, Any]) -> List[str]:
        """Run model generation and decode responses."""
        with torch.no_grad():
            text_ids, _ = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                return_audio=False,
            )

        generated_ids = text_ids[:, inputs["input_ids"].shape[1] :]
        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return responses

    def predict_batch_internal(self, waveforms: List[np.ndarray]) -> List[str]:
        """Run inference on a batch of waveforms."""
        inputs = self.tokenize(waveforms)
        processed_inputs = self.process_hf_inputs(inputs)
        return self.generate_responses(processed_inputs)

    def _parse_instruments(self, prediction: str) -> Dict[str, Any]:
        """Parse the model output to extract instrument list."""
        import ast

        try:
            start = prediction.find("[")
            end = prediction.rfind("]") + 1
            if start != -1 and end > start:
                instruments = ast.literal_eval(prediction[start:end])
                return {"instruments": instruments}
        except Exception:
            pass
        return {"instruments": [prediction.strip()]}

    def _get_waveforms_from_items(
        self, items: List[Dict[str, Any]], now: int
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[np.ndarray]]:
        """Extract waveforms from items, separating valid from errored."""
        import ray

        results = []
        valid_items = []
        valid_waveforms = []

        for item in items:
            # Check if item has an error from preprocessing
            if item.get("error"):
                results.append(self._create_error_result(item, now, item["error"]))
                continue

            # Get waveform - handle both waveform_ref (ObjectRef) and inline waveform
            try:
                if "waveform_ref" in item and item["waveform_ref"] is not None:
                    waveform = ray.get(item["waveform_ref"])
                elif "waveform" in item:
                    waveform = item["waveform"]
                else:
                    raise ValueError("No waveform or waveform_ref in item")

                valid_items.append(item)
                valid_waveforms.append(waveform)

            except Exception as e:
                logger.error(f"Failed to get waveform for {item.get('filename')}: {e}")
                results.append(
                    self._create_error_result(
                        item, now, f"Failed to retrieve waveform: {str(e)}"
                    )
                )

        return results, valid_items, valid_waveforms

    def _create_error_result(
        self, item: Dict[str, Any], now: int, error: str
    ) -> Dict[str, Any]:
        """Create an error result for an item."""
        return {
            "job_id": item["job_id"],
            "song_id": item["song_id"],
            "song_hash": item["song_hash"],
            "filename": item["filename"],
            "instruments": [],
            "detected_at": now,
            "error": error,
        }

    def _create_success_result(
        self, item: Dict[str, Any], now: int, parsed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a success result for an item."""
        return {
            "job_id": item["job_id"],
            "song_id": item["song_id"],
            "song_hash": item["song_hash"],
            "filename": item["filename"],
            "instruments": parsed.get("instruments", []),
            "detected_at": now,
            "error": None,
        }

    def process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of preprocessed audio through the detector."""
        if not items:
            return []

        now = int(time.time())

        # Extract waveforms, separating errors
        results, valid_items, valid_waveforms = self._get_waveforms_from_items(
            items, now
        )

        # Run inference on valid items
        if valid_waveforms:
            inference_start = time.time()
            predictions = self.predict_batch_internal(valid_waveforms)
            inference_time_ms = (time.time() - inference_start) * 1000

            # Update metrics
            self.total_inference_time_ms += inference_time_ms
            self.batch_count += 1
            self.processed_count += len(valid_items)

            logger.info(
                f"Batch inference: {inference_time_ms:.1f}ms "
                f"({inference_time_ms/len(valid_items):.1f}ms/example)"
            )

            # Create results for valid items
            for item, prediction in zip(valid_items, predictions):
                parsed = self._parse_instruments(prediction)
                results.append(self._create_success_result(item, now, parsed))

                logger.debug(f"Detected instruments for {item['filename']}: {parsed}")

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


class InstrumentDetectorCoTAgent(InstrumentDetectorAgent):
    """
    Agent that runs chain-of-thought instrument detection on preprocessed audio.

    Two-step approach:
    1. Planning: Describe sounds in background/middle-ground/foreground layers
    2. Classification: Given descriptions, output structured JSON with instruments

    Uses QwenOmniCoTDataset for conversation building.

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
            "detected_at": int,
        }
    """

    # Override dataset class for CoT prompts
    DATASET_CLS = QwenOmniCoTDataset

    def predict_batch_internal(self, waveforms: List[np.ndarray]) -> List[str]:
        """Run two-step CoT inference on a batch of waveforms."""
        # Step 1: Get layer descriptions
        inputs = self.tokenize(waveforms)
        processed_inputs = self.process_hf_inputs(inputs)

        with torch.no_grad():
            text_ids, _ = self.model.generate(
                **processed_inputs,
                max_new_tokens=512,
                do_sample=False,
                return_audio=False,
            )

        generated_ids = text_ids[:, processed_inputs["input_ids"].shape[1] :]
        step_1_responses = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Step 2: Build step 2 conversations and generate
        conversations = [
            self.DATASET_CLS.get_step_2_conversation(waveform, response)
            for waveform, response in zip(waveforms, step_1_responses)
        ]

        inputs_step_2 = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        processed_inputs_2 = self.process_hf_inputs(inputs_step_2)

        with torch.no_grad():
            text_ids_2, _ = self.model.generate(
                **processed_inputs_2,
                max_new_tokens=512,
                do_sample=False,
                return_audio=False,
            )

        generated_ids_2 = text_ids_2[:, processed_inputs_2["input_ids"].shape[1] :]
        step_2_responses = self.processor.batch_decode(
            generated_ids_2, skip_special_tokens=True
        )

        return step_2_responses

    def _parse_instruments(self, prediction: str) -> Dict[str, Any]:
        """Parse the model output to extract layered instrument JSON."""
        try:
            start = prediction.find("{")
            end = prediction.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(prediction[start:end])

                # Extract layers
                background = parsed.get("background", [])
                middle_ground = parsed.get("middle_ground", [])
                foreground = parsed.get("foreground", [])

                # Flatten for backwards compat
                all_instruments = list(
                    set(background) | set(middle_ground) | set(foreground)
                )

                return {
                    "background": background,
                    "middle_ground": middle_ground,
                    "foreground": foreground,
                    "instruments": all_instruments,
                }
        except Exception as e:
            logger.warning(f"Failed to parse CoT response: {e}")

        return {
            "background": [],
            "middle_ground": [],
            "foreground": [],
            "instruments": [],
        }

    def _create_error_result(
        self, item: Dict[str, Any], now: int, error: str
    ) -> Dict[str, Any]:
        """Create an error result for an item with CoT fields."""
        return {
            "job_id": item["job_id"],
            "song_id": item["song_id"],
            "song_hash": item["song_hash"],
            "filename": item["filename"],
            "background": [],
            "middle_ground": [],
            "foreground": [],
            "instruments": [],
            "detected_at": now,
            "error": error,
        }

    def _create_success_result(
        self, item: Dict[str, Any], now: int, parsed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a success result for an item with CoT fields."""
        return {
            "job_id": item["job_id"],
            "song_id": item["song_id"],
            "song_hash": item["song_hash"],
            "filename": item["filename"],
            "background": parsed.get("background", []),
            "middle_ground": parsed.get("middle_ground", []),
            "foreground": parsed.get("foreground", []),
            "instruments": parsed.get("instruments", []),
            "detected_at": now,
            "error": None,
        }
