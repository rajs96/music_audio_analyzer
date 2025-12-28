import torch
import numpy as np
from typing import List, Dict, Any, Optional

from src.data.qwen_omni import QwenOmniDataset, QwenOmniCoTDataset


class QwenOmniInstrumentDetector:
    """
    Base Qwen-based instrument detector.

    Uses simple single-step prompting with fixed instrument labels.
    Prompts and conversation builders are defined in QwenOmniDataset.
    """

    # Reference dataset class for prompts and conversation building
    DATASET_CLS = QwenOmniDataset

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.model = None
        self.processor = None

        self.load()

    def load(self) -> None:
        """Load the model and processor."""
        from src.models.load_qwen import load_model_and_processor

        self.model, self.processor = load_model_and_processor(
            self.model_name, self.dtype, self.device
        )
        self.model.eval()

    def unload(self) -> None:
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate(
        self, inputs: Dict[str, Any], generate_kwargs: Dict[str, Any] = {}
    ) -> List[str]:
        """Single-step generation for instrument detection."""
        text_ids, _ = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = text_ids[:, inputs["input_ids"].shape[1] :]
        responses: List[str] = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return responses


class QwenOmniCoTInstrumentDetector(QwenOmniInstrumentDetector):
    """
    Qwen-based instrument detector using chain-of-thought prompting.

    Two-step approach:
    1. Planning: Describe sounds in background/middle-ground/foreground layers
    2. Classification: Given descriptions, output structured JSON with instruments

    Prompts and conversation builders are defined in QwenOmniCoTDataset.
    """

    # Override dataset class for CoT prompts
    DATASET_CLS = QwenOmniCoTDataset

    def generate(
        self,
        waveforms: List[np.ndarray],
        inputs: Dict[str, Any],
        planning_generate_kwargs: Optional[Dict[str, Any]] = None,
        response_generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[str], List[str]]:
        """
        Two-step chain-of-thought generation for instrument detection.

        Step 1: Describe what's heard in each layer (uses planning_generate_kwargs)
        Step 2: Convert descriptions to structured JSON (uses response_generate_kwargs)

        Args:
            waveforms: List of audio waveforms for step 2 conversation building
            inputs: Tokenized inputs for step 1
            planning_generate_kwargs: Generation kwargs for step 1 (planning/description)
            response_generate_kwargs: Generation kwargs for step 2 (JSON response)

        Returns:
            Tuple of (planning_responses, final_responses)
        """
        planning_kwargs = planning_generate_kwargs or {}
        response_kwargs = response_generate_kwargs or {}

        # Step 1: Get layer descriptions
        step_1_responses = super().generate(inputs, planning_kwargs)

        # Step 2: Build conversations using dataset's conversation builder
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
        inputs_step_2 = {
            k: v.to(self.device, dtype=self.dtype) for k, v in inputs_step_2.items()
        }

        step_2_responses = super().generate(inputs_step_2, response_kwargs)

        return step_1_responses, step_2_responses
