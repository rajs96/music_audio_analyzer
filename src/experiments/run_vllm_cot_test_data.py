"""
Test script for vLLM-based CoT instrument detection.

Uses vLLM for efficient batched inference via use_vllm=True flag.
"""

import json
from tqdm import tqdm
from loguru import logger
import time
from pathlib import Path
import pandas as pd
import argparse
import torchaudio
import numpy as np

from src.models.qwen_instrument_detector import QwenOmniCoTInstrumentDetector


DEFAULT_PLANNING_KWARGS = {
    "temperature": 0.0,
    "max_tokens": 1024,
}

DEFAULT_RESPONSE_KWARGS = {
    "temperature": 0.0,
    "max_tokens": 256,
}


def decode_audio(filepath: str, target_sr: int = 16000) -> np.ndarray:
    """Decode audio file to waveform at target sample rate."""
    wav, sr = torchaudio.load(filepath)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)

    return wav.squeeze(0).numpy()


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


def load_audio_files(data_dir: str) -> list:
    """Load all audio files from directory."""
    audio_files = []
    data_path = Path(data_dir)

    for ext in ["*.mp3", "*.wav", "*.flac"]:
        audio_files.extend(data_path.glob(f"**/{ext}"))

    return sorted(audio_files)


def main(args):
    model_name = args.model_name
    data_dir = args.data_dir
    results_dir = Path(args.results_dir).resolve()
    logger.info(f"Results directory: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    target_sr = 16000

    logger.info(f"Loading vLLM model: {model_name}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size or 'auto'}")

    # Load detector with vLLM backend
    detector = QwenOmniCoTInstrumentDetector(
        model_name=model_name,
        use_vllm=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    logger.info("vLLM detector loaded")

    # Load audio files
    audio_files = load_audio_files(data_dir)
    logger.info(f"Found {len(audio_files)} audio files")

    if not audio_files:
        logger.error(f"No audio files found in {data_dir}")
        return

    # Process in batches
    results = []
    start_time = time.time()
    total_processed = 0

    # Create batches
    batches = [
        audio_files[i : i + batch_size] for i in range(0, len(audio_files), batch_size)
    ]

    for batch_idx, batch_files in tqdm(enumerate(batches), total=len(batches)):
        logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}")

        # Load and decode audio
        filenames = [str(f) for f in batch_files]
        waveforms = [decode_audio(f, target_sr) for f in filenames]

        # Generate using vLLM CoT detector
        try:
            planning_responses, final_responses = detector.generate(
                waveforms=waveforms,
                planning_sampling_kwargs=DEFAULT_PLANNING_KWARGS,
                response_sampling_kwargs=DEFAULT_RESPONSE_KWARGS,
            )
        except Exception as e:
            logger.error(f"Error generating: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Parse and collect results
        for filename, planning_response, final_response in zip(
            filenames, planning_responses, final_responses
        ):
            song_name = Path(filename).stem
            instruments = parse_instruments_json(final_response)

            results.append(
                {
                    "filename": filename,
                    "song_name": song_name,
                    "background": instruments.get("background", []),
                    "middle_ground": instruments.get("middle_ground", []),
                    "foreground": instruments.get("foreground", []),
                    "planning_response": planning_response,
                    "raw_response": final_response,
                }
            )

            # log every N examples
            if total_processed % 100 == 0:
                logger.info(f"Example {total_processed}:")
                logger.info(f"{song_name}:")
                logger.info(f"Planning response: {planning_response[:200]}...")
                logger.info(f"Final response: {final_response}")
                logger.info(f"  Background: {instruments.get('background', [])}")
                logger.info(f"  Middle-ground: {instruments.get('middle_ground', [])}")
                logger.info(f"  Foreground: {instruments.get('foreground', [])}")

        total_processed += len(filenames)

    # Save results
    results_df = pd.DataFrame(results)
    output_file = results_dir / "results_vllm_cot.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

    end_time = time.time()
    total_files = len(results)
    elapsed = end_time - start_time

    logger.info(f"Time taken: {elapsed:.1f} seconds")
    logger.info(f"Total files processed: {total_files}")
    if total_files > 0:
        logger.info(f"Time per file: {elapsed / total_files:.2f} seconds")
        logger.info(f"Throughput: {total_files / elapsed:.2f} files/second")

    detector.unload()
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run vLLM-based CoT instrument detection"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="eval_audio_files",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/app/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism (default: auto)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization (0.0-1.0)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=8,
        help="Maximum number of sequences to process in parallel",
    )

    args = parser.parse_args()
    main(args)
