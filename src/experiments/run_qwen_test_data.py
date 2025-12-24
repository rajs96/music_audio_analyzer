import torch
from torch.utils.data import DataLoader
from loguru import logger
import time
import os
from src.instrument_detect.models.load_qwen import load_model_and_processor
from src.instrument_detect.data.qwen_omni import QwenOmniDataset


def main():
    model_name = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    data_dir = "audio_files"
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}")

    # Load model and processor
    model, processor = load_model_and_processor(model_name, device)
    model.eval()
    logger.info("Model loaded")

    # Create dataset
    dataset = QwenOmniDataset(data_dir, processor)
    logger.info(f"Dataset size: {len(dataset)} files")

    # Create dataloader with prefetching and multiple workers
    num_workers = os.cpu_count()  # Parallel audio decoding
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=True,  # Keep workers alive between batches
        collate_fn=lambda x: x[0],  # Dataset already returns batched inputs
    )
    logger.info(f"DataLoader: {num_workers} workers, prefetch_factor=2")

    results = []

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            # Move inputs to device
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            # Generate
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

            # Decode output
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)

            logger.info(f"Response: {response}")
            results.append(response)

    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time} seconds")
    logger.info(
        f"Time taken per file: {(end_time - start_time) / len(dataloader)} seconds"
    )

    logger.info("--- All Results ---")
    for i, result in enumerate(results):
        logger.info(f"File {i + 1}: {result}")

    return results


if __name__ == "__main__":
    main()
