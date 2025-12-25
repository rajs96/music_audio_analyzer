import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor


def load_model_and_processor(model_name: str, device: str):
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map=device,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
    return model, processor
