"""Pre-download and save Qwen model and processor to disk."""

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
MODEL_CACHE_PATH = "/app/cache/models"
PROCESSOR_CACHE_PATH = "/app/cache/processor"


def main():
    print(f"Downloading model: {MODEL_NAME}")

    # Download and save processor
    print("Saving processor...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_NAME)
    processor.save_pretrained(PROCESSOR_CACHE_PATH)
    print(f"Processor saved to {PROCESSOR_CACHE_PATH}")

    # Download and save model
    print("Saving model (this may take a while)...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        attn_implementation="flash_attention_2",
    )
    model.save_pretrained(MODEL_CACHE_PATH)
    print(f"Model saved to {MODEL_CACHE_PATH}")

    print("Done!")


if __name__ == "__main__":
    main()
