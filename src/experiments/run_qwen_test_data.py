import ast
import torch
from torch.utils.data import DataLoader
from loguru import logger
import time
import os
from typing import List
from pathlib import Path
import pandas as pd
from collections import defaultdict
from src.models import load_model_and_processor
from src.data import QwenOmniDataset


def identity_collate(x):
    """Collate function that returns first element (dataset already batches)."""
    return x[0]


def parse_instruments(response: str) -> set:
    """Parse instrument list from model response string."""
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            instruments = ast.literal_eval(response[start:end])
            return set(instruments)
    except Exception:
        pass
    return set()


def inter_response_agreement(responses: List[str]) -> float:
    """Calculate inter-response agreement between responses."""
    all_response_strings: list[set[str]] = [
        set(parse_instruments(r)) for r in responses
    ]
    agreement_strings = set.intersection(*all_response_strings)
    max_length = max(len(r) for r in all_response_strings)
    return len(agreement_strings) / max_length


def main():
    model_name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    data_dir = "audio_files"
    results_dir = Path("/app/results").resolve()
    logger.info(f"Results directory: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    batch_size = 4
    num_ensemble_runs = 3  # Number of generations to ensemble
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Track all responses per file for ensembling
    file_responses = defaultdict(list)  # filename -> list of responses
    file_instruments = defaultdict(set)  # filename -> union of instruments

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Ensemble runs: {num_ensemble_runs}")

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
        for ensemble_idx in range(num_ensemble_runs):
            logger.info(f"=== Ensemble run {ensemble_idx + 1}/{num_ensemble_runs} ===")

            for batch_idx, (filenames, inputs) in enumerate(dataloader):
                logger.info(
                    f"Processing batch {batch_idx + 1}/{len(dataloader)} "
                    f"(ensemble {ensemble_idx + 1}/{num_ensemble_runs})"
                )

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

                # Generate
                try:
                    text_ids, _ = model.generate(
                        **processed_inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.2,
                        top_p=0.9,
                        top_k=50,
                        return_audio=False,
                    )
                except Exception as e:
                    logger.error(f"Error generating: {e}")
                    continue

                # Decode output
                generated_ids = text_ids[:, processed_inputs["input_ids"].shape[1] :]
                responses = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                # Collect responses and parse instruments
                for filename, response in zip(filenames, responses):
                    file_responses[filename].append(response)
                    instruments = parse_instruments(response)
                    file_instruments[filename].update(instruments)
                    logger.debug(f"{filename} run {ensemble_idx + 1}: {instruments}")

    # Build final results with ensembled instruments
    results = []
    for filename in file_responses.keys():
        song_name = filename.split("/")[-1].split(".")[0]
        all_responses = file_responses[filename]  # this is a list of strings
        ensembled_instruments = sorted(list(file_instruments[filename]))

        results.append(
            {
                "filename": filename,
                "song_name": song_name,
                "ensembled_instruments": ensembled_instruments,
                "individual_responses": all_responses,
                "num_runs": len(all_responses),
            }
        )
        logger.info(
            f"{song_name}: {ensembled_instruments} (from {len(all_responses)} runs)"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "results.csv", index=False)
    end_time = time.time()
    total_generations = len(dataloader) * batch_size * num_ensemble_runs
    logger.info(f"Time taken: {end_time - start_time:.1f} seconds")
    logger.info(f"Total generations: {total_generations}")
    logger.info(
        f"Time per generation: {(end_time - start_time) / total_generations:.2f} seconds"
    )

    logger.info("--- Ensembled Results ---")
    for _, result in results_df.iterrows():
        logger.info(f"Song {result.song_name}: {result.ensembled_instruments}")

    logger.info(results_df.head(20))


if __name__ == "__main__":
    main()
