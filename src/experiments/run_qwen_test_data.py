import torch
from torch.utils.data import DataLoader
from loguru import logger
import time
import os
from pathlib import Path
import pandas as pd
from src.pipelines.instrument_detection.models import load_model_and_processor
from src.pipelines.instrument_detection.data import QwenOmniDataset


def identity_collate(x):
    """Collate function that returns first element (dataset already batches)."""
    return x[0]


def main():
    model_name = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    data_dir = "audio_files"
    # results_dir = Path("/app/results").resolve()
    # logger.info(f"Results directory: {results_dir}")
    # results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    batch_size = 4
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}")

    # Load model and processor
    model, processor = load_model_and_processor(model_name, dtype, device)
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
        collate_fn=identity_collate,  # Dataset already returns batched inputs
    )
    logger.info(f"DataLoader: {num_workers} workers, prefetch_factor=2")

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (filenames, inputs) in enumerate(dataloader):

            logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            # Move inputs to device
            processed_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point():
                        processed_inputs[k] = v.to(device).to(dtype)
                    else:
                        processed_inputs[k] = v.to(device)
                else:
                    processed_inputs[k] = v

            logger.info(f"processed: {processed_inputs}")
            # Generate
            try:
                output_ids = model.generate(
                    **processed_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
            except Exception as e:
                logger.error(f"Error generating: {e}")
                continue

            # Decode output
            generated_ids = output_ids[:, processed_inputs["input_ids"].shape[1] :]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)

            logger.info(f"Response: {response}")
            for filename, response in zip(filenames, response):
                song_name = filename.split("/")[-1].split(".")[0]
                results.append(
                    {
                        "filename": filename,
                        "song_name": song_name,
                        "response": response,
                    }
                )

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "results.csv", index=False)
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time} seconds")
    logger.info(
        f"Time taken per file: {(end_time - start_time) / len(dataloader)} seconds"
    )

    logger.info("--- All Results ---")
    for result in results.iterrows():
        logger.info(f"Song {result.song_name}: {result.response}")

    logger.info(results_df.head(20))


if __name__ == "__main__":
    main()
