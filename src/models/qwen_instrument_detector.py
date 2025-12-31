"""
Qwen-based instrument detector with support for both HuggingFace and vLLM backends.

Use `use_vllm=True` for efficient batched inference with vLLM.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional

from src.data.qwen_omni import QwenOmniDataset, QwenOmniCoTDataset

# from qwen docs
os.environ["VLLM_USE_V1"] = "0"


class QwenOmniInstrumentDetector:
    """
    Base Qwen-based instrument detector.

    Supports both HuggingFace transformers and vLLM backends.
    Use `use_vllm=True` for efficient batched inference.

    Prompts and conversation builders are defined in QwenOmniDataset.
    """

    # Reference dataset class for prompts and conversation building
    DATASET_CLS = QwenOmniDataset

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        use_vllm: bool = False,
        tensor_parallel_size: Optional[int] = None,
        pipeline_parallel_size: int = 1,
        distributed_executor_backend: str = "mp",
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 32768,
        max_num_seqs: int = 8,
        seed: int = 1234,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.use_vllm = use_vllm

        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.distributed_executor_backend = distributed_executor_backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.seed = seed

        # Will be set by load methods
        self.model = None  # HF model
        self.llm = None  # vLLM engine
        self.processor = None

        self.load()

    def load(self) -> None:
        """Load the model using appropriate backend."""
        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_hf()

    def _load_hf(self) -> None:
        """Load model using HuggingFace transformers."""
        from src.models.load_qwen import load_model_and_processor

        self.model, self.processor = load_model_and_processor(
            self.model_name, self.dtype, self.device
        )
        self.model.eval()

    def _load_vllm(self) -> None:
        """Load model using vLLM engine."""
        from vllm import LLM
        from transformers import Qwen3OmniMoeProcessor
        from loguru import logger

        tp_size = self.tensor_parallel_size or torch.cuda.device_count()
        pp_size = self.pipeline_parallel_size

        logger.info(
            f"Loading vLLM engine: model={self.model_name}, "
            f"tp={tp_size}, pp={pp_size}, backend={self.distributed_executor_backend}"
        )

        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            distributed_executor_backend=self.distributed_executor_backend,
            limit_mm_per_prompt={"audio": 1},  # only one audio per prompt, but batches
            max_num_seqs=self.max_num_seqs,
            max_model_len=self.max_model_len,
            seed=self.seed,
            enforce_eager=True,  # Disable CUDA graphs - more stable, slightly slower
        )

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_name)

    def unload(self) -> None:
        """Unload model and free memory."""
        if self.use_vllm:
            if self.llm is not None:
                del self.llm
                self.llm = None
        else:
            if self.model is not None:
                del self.model
                self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _process_hf_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Move HF inputs to device with correct dtype."""
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    # there are longs that need stay that way
                    processed_inputs[k] = v.to(self.device, dtype=self.dtype)
                else:
                    processed_inputs[k] = v.to(self.device)
            else:
                processed_inputs[k] = v
        return processed_inputs

    def _generate_hf(
        self, inputs: Dict[str, Any], generate_kwargs: Dict[str, Any]
    ) -> List[str]:
        """Generate using HuggingFace transformers."""
        processed_inputs = self._process_hf_inputs(inputs)
        text_ids, _ = self.model.generate(**processed_inputs, **generate_kwargs)
        generated_ids = text_ids[:, inputs["input_ids"].shape[1] :]
        responses: List[str] = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return responses

    def _build_vllm_input(
        self, waveform: np.ndarray, conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build a single vLLM input dict from a waveform and conversation."""
        prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {
            "prompt": prompt,
            "multi_modal_data": {
                "audio": [waveform],
            },
            "mm_processor_kwargs": {},
        }

    def _build_vllm_inputs(self, waveforms: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Build vLLM inputs for a batch of waveforms."""
        inputs = []
        for waveform in waveforms:
            conversation = self.DATASET_CLS.get_conversation(waveform)
            inputs.append(self._build_vllm_input(waveform, conversation))
        return inputs

    def _generate_vllm(
        self, waveforms: List[np.ndarray], sampling_kwargs: Dict[str, Any]
    ) -> List[str]:
        """Generate using vLLM engine."""
        from vllm import SamplingParams

        inputs = self._build_vllm_inputs(waveforms)
        sampling_params = SamplingParams(**sampling_kwargs)
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        return responses

    def generate(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        waveforms: Optional[List[np.ndarray]] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate instrument predictions.

        For HuggingFace backend:
            - Pass `inputs` (tokenized HF inputs) and optionally `generate_kwargs`

        For vLLM backend:
            - Pass `waveforms` (list of numpy arrays) and optionally `sampling_kwargs`

        Args:
            inputs: Tokenized HF inputs (for HF backend)
            waveforms: List of audio waveforms (for vLLM backend)
            generate_kwargs: HF generate kwargs (max_new_tokens, do_sample, etc.)
            sampling_kwargs: vLLM SamplingParams kwargs (temperature, max_tokens, etc.)

        Returns:
            List of model response strings
        """
        if self.use_vllm:
            if waveforms is None:
                raise ValueError("waveforms required for vLLM backend")
            kwargs = sampling_kwargs or {"temperature": 0.0, "max_tokens": 256}
            return self._generate_vllm(waveforms, kwargs)
        else:
            if inputs is None:
                raise ValueError("inputs required for HuggingFace backend")
            kwargs = generate_kwargs or {}
            return self._generate_hf(inputs, kwargs)


class QwenOmniCoTInstrumentDetector(QwenOmniInstrumentDetector):
    """
    Qwen-based instrument detector using chain-of-thought prompting.

    Supports both HuggingFace transformers and vLLM backends.

    Two-step approach:
    1. Planning: Describe sounds in background/middle-ground/foreground layers
    2. Detection: Given descriptions, output structured JSON with instruments

    Prompts and conversation builders are defined in QwenOmniCoTDataset.
    """

    # Override dataset class for CoT prompts
    DATASET_CLS = QwenOmniCoTDataset

    # ==================== HuggingFace CoT Generate ====================

    def _generate_cot_hf(
        self,
        waveforms: List[np.ndarray],
        inputs: Dict[str, Any],
        planning_kwargs: Dict[str, Any],
        response_kwargs: Dict[str, Any],
    ) -> tuple[List[str], List[str]]:
        """Two-step CoT generation using HuggingFace."""
        # Step 1: Get layer descriptions
        step_1_responses = self._generate_hf(inputs, planning_kwargs)

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

        step_2_responses = self._generate_hf(inputs_step_2, response_kwargs)

        return step_1_responses, step_2_responses

    # ==================== vLLM CoT Generate ====================

    def _generate_cot_vllm(
        self,
        waveforms: List[np.ndarray],
        planning_kwargs: Dict[str, Any],
        response_kwargs: Dict[str, Any],
    ) -> tuple[List[str], List[str]]:
        """Two-step CoT generation using vLLM."""
        from vllm import SamplingParams
        from loguru import logger

        # Step 1: Build inputs for planning
        step_1_inputs = self._build_vllm_inputs(waveforms)

        # Log audio durations for debugging
        for i, wf in enumerate(waveforms):
            duration_sec = len(wf) / 16000  # assuming 16kHz
            logger.debug(f"  Waveform {i}: {duration_sec:.1f}s ({len(wf)} samples)")

        # Generate step 1
        logger.debug(f"Step 1: Generating planning for {len(waveforms)} waveforms...")
        planning_params = SamplingParams(**planning_kwargs)
        step_1_outputs = self.llm.generate(
            step_1_inputs, sampling_params=planning_params
        )
        planning_responses = [output.outputs[0].text for output in step_1_outputs]

        # Log planning response lengths
        for i, resp in enumerate(planning_responses):
            logger.debug(f"  Planning {i}: {len(resp)} chars")

        # Step 2: Build inputs using step 1 responses
        step_2_inputs = []
        for waveform, planning_response in zip(waveforms, planning_responses):
            conversation = self.DATASET_CLS.get_step_2_conversation(
                waveform, planning_response
            )
            step_2_inputs.append(self._build_vllm_input(waveform, conversation))

        # Log step 2 prompt lengths
        for i, inp in enumerate(step_2_inputs):
            logger.debug(f"  Step2 prompt {i}: {len(inp['prompt'])} chars")

        # Generate step 2
        logger.debug(f"Step 2: Generating final responses...")
        response_params = SamplingParams(**response_kwargs)
        step_2_outputs = self.llm.generate(
            step_2_inputs, sampling_params=response_params
        )
        final_responses = [output.outputs[0].text for output in step_2_outputs]

        return planning_responses, final_responses

    # ==================== Public Generate Method ====================

    def generate(
        self,
        waveforms: List[np.ndarray],
        inputs: Optional[Dict[str, Any]] = None,
        # HF kwargs
        planning_generate_kwargs: Optional[Dict[str, Any]] = None,
        response_generate_kwargs: Optional[Dict[str, Any]] = None,
        # vLLM kwargs
        planning_sampling_kwargs: Optional[Dict[str, Any]] = None,
        response_sampling_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[str], List[str]]:
        """
        Two-step chain-of-thought generation for instrument detection.

        For HuggingFace backend:
            - Pass `waveforms`, `inputs`, and optionally `planning_generate_kwargs`/`response_generate_kwargs`

        For vLLM backend:
            - Pass `waveforms` and optionally `planning_sampling_kwargs`/`response_sampling_kwargs`

        Args:
            waveforms: List of audio waveforms (required for both backends)
            inputs: Tokenized HF inputs for step 1 (required for HF backend)
            planning_generate_kwargs: HF generate kwargs for step 1
            response_generate_kwargs: HF generate kwargs for step 2
            planning_sampling_kwargs: vLLM SamplingParams kwargs for step 1
            response_sampling_kwargs: vLLM SamplingParams kwargs for step 2

        Returns:
            Tuple of (planning_responses, final_responses)
        """
        if self.use_vllm:
            planning_kwargs = planning_sampling_kwargs or {
                "temperature": 0.0,
                "max_tokens": 1024,
            }
            response_kwargs = response_sampling_kwargs or {
                "temperature": 0.0,
                "max_tokens": 256,
            }
            return self._generate_cot_vllm(waveforms, planning_kwargs, response_kwargs)
        else:
            if inputs is None:
                raise ValueError("inputs required for HuggingFace backend")
            planning_kwargs = planning_generate_kwargs or {}
            response_kwargs = response_generate_kwargs or {}
            return self._generate_cot_hf(
                waveforms, inputs, planning_kwargs, response_kwargs
            )
