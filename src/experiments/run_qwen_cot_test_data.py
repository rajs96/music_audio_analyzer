import json
import torch
from torch.utils.data import DataLoader
from loguru import logger
import time
import os
from pathlib import Path
import pandas as pd

from src.data import QwenOmniCoTDataset
from src.models.qwen_instrument_detector import QwenOmniCoTInstrumentDetector


def identity_collate(x):
    """Collate function that returns first element (dataset already batches)."""
    return x[0]


def parse_instruments_json(response: str) -> dict:
    """Parse instrument JSON from model response string."""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
    except Exception:
        pass
    return {"background": [], "middle_ground": [], "foreground": []}


def main():
    model_name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    data_dir = "audio_files"
    results_dir = Path("/app/results").resolve()
    logger.info(f"Results directory: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    batch_size = 4
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}")

    # Load detector
    detector = QwenOmniCoTInstrumentDetector(
        model_name=model_name,
        dtype=dtype,
        device=device,
    )
    logger.info("Detector loaded")

    # Create dataset
    dataset = QwenOmniCoTDataset(data_dir, detector.processor)
    logger.info(f"Dataset size: {len(dataset)} files")

    # Create dataloader
    num_workers = os.cpu_count()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=identity_collate,
    )
    logger.info(f"DataLoader: {num_workers} workers, prefetch_factor=2")

    generate_kwargs = {
        "max_new_tokens": 512,
        "do_sample": False,
        "return_audio": False,
    }

    results = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (filenames, waveforms, inputs) in enumerate(dataloader):
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

            # Generate using CoT detector
            try:
                responses = detector.generate(
                    waveforms=waveforms,
                    inputs=processed_inputs,
                    generate_kwargs=generate_kwargs,
                )
            except Exception as e:
                logger.error(f"Error generating: {e}")
                import traceback

                traceback.print_exc()
                continue

            # Parse and collect results
            for filename, response in zip(filenames, responses):
                song_name = filename.split("/")[-1].split(".")[0]
                instruments = parse_instruments_json(response)

                results.append(
                    {
                        "filename": filename,
                        "song_name": song_name,
                        "background": instruments.get("background", []),
                        "middle_ground": instruments.get("middle_ground", []),
                        "foreground": instruments.get("foreground", []),
                        "raw_response": response,
                    }
                )

                logger.info(f"{song_name}:")
                logger.info(f"  Background: {instruments.get('background', [])}")
                logger.info(f"  Middle-ground: {instruments.get('middle_ground', [])}")
                logger.info(f"  Foreground: {instruments.get('foreground', [])}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "results_cot.csv", index=False)

    end_time = time.time()
    total_files = len(results)
    logger.info(f"Time taken: {end_time - start_time:.1f} seconds")
    logger.info(f"Total files processed: {total_files}")
    logger.info(f"Time per file: {(end_time - start_time) / total_files:.2f} seconds")

    detector.unload()
    logger.info("Done!")


if __name__ == "__main__":
    main()
