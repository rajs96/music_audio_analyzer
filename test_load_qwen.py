from transformers import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from loguru import logger

MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
if __name__ == "__main__":
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype="float16",
        device_map="cpu",
    )
    logger.info("Model loaded")
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_NAME)
    logger.info("Processor loaded")
