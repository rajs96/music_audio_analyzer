"""
Instrument Detector Agent.

Runs ML inference to detect instruments in preprocessed audio.
"""

import time
from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger

from src.streaming_pipeline import Agent


class InstrumentDetectorAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Agent that runs instrument detection on preprocessed audio.

    The model is loaded once in setup() and reused for all batches.

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

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        dtype: torch.dtype = torch.float32,
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
        from src.pipelines.instrument_detection.models.load_qwen import (
            load_model_and_processor,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"InstrumentDetectorAgent loading model {self.model_name} "
            f"on {self.device} with dtype={self.dtype}"
        )

        self.model, self.processor = load_model_and_processor(
            self.model_name, self.dtype, self.device
        )
        self.model.eval()
        logger.info("InstrumentDetectorAgent model loaded")

    def teardown(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("InstrumentDetectorAgent cleaned up")

    def get_system_prompt(self) -> Dict[str, Any]:
        text = """
        You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.
        You are also an expert in detecting what instruments are being played in a song.

        You will be given a song and you will need to detect what instruments are being played in the song.
        Return a list of strings, each string is the name of an instrument.

        Only use the following strings:
        - drums
        - bass
        - electric_guitar
        - acoustic_guitar
        - piano
        - synthesizer
        - strings
        - wind
        - lead_vocals
        - backing_vocals

        Example output 1: ['drums', 'electric_guitar', 'piano', 'lead_vocals']
        Example output 2: ['acoustic_guitar', 'piano', 'backing_vocals', 'bass']
        Example output 3: ['piano', 'lead_vocals']
        """
        return {
            "role": "system",
            "content": [{"type": "text", "text": text}],
        }

    def get_user_prompt(self, waveform: np.ndarray) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [{"type": "audio", "audio": waveform}],
        }

    def tokenize(self, waveforms: List[np.ndarray]):
        """Convert waveforms to model inputs."""
        conversations = []
        for waveform in waveforms:
            conversation = [self.get_system_prompt(), self.get_user_prompt(waveform)]
            conversations.append(conversation)

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return inputs

    def predict_batch_internal(self, waveforms: List[np.ndarray]) -> List[str]:
        """Run inference on a batch of waveforms."""
        inputs = self.tokenize(waveforms)
        inputs = {
            k: (
                v.to(
                    device=self.device,
                    dtype=self.dtype if v.is_floating_point() else None,
                )
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return responses

    def _parse_instruments(self, prediction: str) -> List[str]:
        """Parse the model output to extract instrument list."""
        try:
            import ast

            start = prediction.find("[")
            end = prediction.rfind("]") + 1
            if start != -1 and end > start:
                return ast.literal_eval(prediction[start:end])
        except Exception:
            pass
        return [prediction.strip()]

    def process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of preprocessed audio through the detector."""
        import ray

        if not items:
            return []

        now = int(time.time())
        results = []

        # Separate valid items from errored items
        valid_items = []
        valid_waveforms = []

        for item in items:
            # Check if item has an error from preprocessing
            if item.get("error"):
                # Propagate error through the pipeline
                results.append(
                    {
                        "job_id": item["job_id"],
                        "song_id": item["song_id"],
                        "song_hash": item["song_hash"],
                        "filename": item["filename"],
                        "instruments": [],
                        "detected_at": now,
                        "error": item["error"],
                    }
                )
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
                    {
                        "job_id": item["job_id"],
                        "song_id": item["song_id"],
                        "song_hash": item["song_hash"],
                        "filename": item["filename"],
                        "instruments": [],
                        "detected_at": now,
                        "error": f"Failed to retrieve waveform: {str(e)}",
                    }
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
                instruments = self._parse_instruments(prediction)

                results.append(
                    {
                        "job_id": item["job_id"],
                        "song_id": item["song_id"],
                        "song_hash": item["song_hash"],
                        "filename": item["filename"],
                        "instruments": instruments,
                        "detected_at": now,
                        "error": None,
                    }
                )

                logger.debug(
                    f"Detected instruments for {item['filename']}: {instruments}"
                )

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
