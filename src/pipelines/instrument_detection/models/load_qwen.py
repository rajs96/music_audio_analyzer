import os
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from loguru import logger


def load_model_and_processor(model_name: str, dtype: torch.dtype, device: str):
    # When Ray assigns a GPU, it sets CUDA_VISIBLE_DEVICES so only that GPU is visible.
    # Use cuda:0 to load onto the first (and only) visible GPU for this actor.
    if device == "cuda":
        device = "cuda:0"

    logger.info(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
    )
    logger.info(f"Loading model to device={device}, dtype={dtype}")

    # Load model without device_map to avoid accelerate dependency,
    # then manually move to device
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
    )
    model = model.to(device)
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
    return model, processor
