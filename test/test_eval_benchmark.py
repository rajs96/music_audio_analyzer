"""
Benchmark test on eval_audio_files.

Processes all files in eval_audio_files and reports:
- Model loading time (separate)
- Pure inference time
- Throughput (files/second) based on inference only

Run with: python test/test_eval_benchmark.py
"""

import hashlib
import json
import signal
import threading
import time
import uuid
from pathlib import Path

import ray
import torch
from loguru import logger
from ray.util.queue import Queue

from src.streaming_pipeline import (
    AgentRayComputeConfig,
    AgentStage,
    QueueStreamingDatasource,
    StreamingDatasourceConfig,
    StreamingPipeline,
)
from src.pipelines.instrument_detection.agents.audio_preprocessor import (
    AudioPreprocessorAgent,
)
from src.pipelines.instrument_detection.agents.instrument_detector import (
    InstrumentDetectorCoTAgent,
)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

DEFAULT_MODELS_DIR = "/app/models"


def download_model_for_vllm(
    model_name: str, models_dir: str = DEFAULT_MODELS_DIR
) -> str:
    from huggingface_hub import snapshot_download

    model_cache_name = model_name.replace("/", "_")
    local_path = Path(models_dir) / model_cache_name

    if local_path.exists() and any(local_path.iterdir()):
        logger.info(f"Model already downloaded at {local_path}")
        return str(local_path)

    logger.info(f"Downloading model {model_name} to {local_path}...")
    local_path.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
    )

    logger.info(f"Model downloaded successfully to {local_path}")
    return str(local_path)


def create_job_from_file(filepath: Path) -> dict:
    audio_bytes = filepath.read_bytes()
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    song_id = f"trk_{uuid.uuid4().hex[:12]}"
    song_hash = hashlib.sha256(audio_bytes).hexdigest()

    return {
        "job_id": job_id,
        "song_id": song_id,
        "song_hash": song_hash,
        "filename": filepath.name,
        "audio_bytes": audio_bytes,
    }


class ResultCollector(threading.Thread):
    def __init__(self, streaming_iterator, total_expected: int):
        super().__init__(daemon=True)
        self.streaming_iterator = streaming_iterator
        self.total_expected = total_expected
        self.results = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.first_result_time = None

    def run(self):
        logger.info("ResultCollector started")
        try:
            for batch in self.streaming_iterator:
                if self.stop_event.is_set():
                    break

                if batch:
                    keys = list(batch.keys())
                    if keys:
                        n_items = len(batch[keys[0]])
                        for i in range(n_items):
                            result = {k: batch[k][i] for k in keys}

                            with self.lock:
                                if self.first_result_time is None:
                                    self.first_result_time = time.time()
                                self.results.append(result)
                                count = len(self.results)

                            if result.get("error") and str(result.get("error")).strip():
                                logger.warning(
                                    f"[{count}] {result['filename']} -> ERROR"
                                )
                            else:
                                logger.info(
                                    f"[{count}/{self.total_expected}] {result['filename']}"
                                )

                with self.lock:
                    if len(self.results) >= self.total_expected:
                        break

        except Exception as e:
            logger.error(f"ResultCollector error: {e}")
            import traceback

            traceback.print_exc()

        logger.info(f"ResultCollector finished with {len(self.results)} results")

    def stop(self):
        self.stop_event.set()

    def get_results(self):
        with self.lock:
            return list(self.results)

    def get_count(self):
        with self.lock:
            return len(self.results)


def main(
    data_dir: str = "eval_audio_files",
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    models_dir: str = DEFAULT_MODELS_DIR,
    dtype: torch.dtype = torch.bfloat16,
    tensor_parallel_size: int = 1,
    distributed_executor_backend: str = "mp",
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 32768,
    max_num_seqs: int = 8,
    detector_actors: int = 1,
    detector_batch_size: int = 4,
    preprocessor_actors: int = 4,
    preprocessor_batch_size: int = 8,
    output_file: str = None,
):
    # Download model
    logger.info("=" * 60)
    logger.info("DOWNLOADING MODEL")
    logger.info("=" * 60)
    model_download_start = time.time()
    model_path = download_model_for_vllm(model_name, models_dir)
    model_download_time = time.time() - model_download_start
    logger.info(f"Model download/check: {model_download_time:.1f}s")

    # Get audio files
    audio_dir = Path(__file__).parent.parent / data_dir
    audio_files = sorted([f for f in audio_dir.glob("*.mp3") if f.is_file()])
    total_jobs = len(audio_files)

    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        return 1

    logger.info(f"Found {total_jobs} audio files in {audio_dir}")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Create job queue and submit all jobs upfront
    job_queue = Queue(maxsize=total_jobs + 100)

    logger.info("Loading all audio files into queue...")
    file_load_start = time.time()
    for audio_file in audio_files:
        job = create_job_from_file(audio_file)
        job_queue.put(job)
    file_load_time = time.time() - file_load_start
    logger.info(f"Loaded {total_jobs} files in {file_load_time:.1f}s")

    # Create datasource
    datasource = QueueStreamingDatasource(
        queue=job_queue,
        item_to_row_fn=lambda x: x,
        config=StreamingDatasourceConfig(
            parallelism=1,
            batch_size=8,
            batch_timeout=0.1,
            poll_interval=0.01,
            max_items=total_jobs,
        ),
    )

    # Create pipeline stages
    preprocessor_stage = AgentStage(
        agent=AudioPreprocessorAgent(target_sr=16000),
        config=AgentRayComputeConfig(
            num_actors=preprocessor_actors,
            batch_size=preprocessor_batch_size,
            num_cpus=1.0,
        ),
        name="AudioPreprocessor",
    )

    detector_stage = AgentStage(
        agent=InstrumentDetectorCoTAgent(
            model_name=model_path,
            dtype=dtype,
            use_vllm=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        ),
        config=AgentRayComputeConfig(
            num_actors=detector_actors,
            batch_size=detector_batch_size,
            num_gpus=float(tensor_parallel_size),
            tensor_parallel_size=tensor_parallel_size,
        ),
        name="InstrumentDetector",
    )

    pipeline = StreamingPipeline(
        datasource=datasource,
        stages=[preprocessor_stage, detector_stage],
        name="EvalBenchmarkPipeline",
    )

    # Shutdown handling
    shutdown_requested = threading.Event()

    def signal_handler(signum, _frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_requested.set()
        datasource.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Warmup
    def get_warmup_data():
        warmup_file = audio_files[0]
        audio_bytes = warmup_file.read_bytes()
        return [
            {
                "job_id": "warmup_001",
                "song_id": "warmup",
                "song_hash": "warmup",
                "filename": warmup_file.name,
                "audio_bytes": audio_bytes,
            }
        ]

    logger.info("=" * 60)
    logger.info("LOADING MODEL (warmup)")
    logger.info("=" * 60)
    model_load_start = time.time()
    streaming_iterator = pipeline.warmup_and_stream(
        warmup_data_fn=get_warmup_data,
        warmup_timeout=300.0,
        batch_size=1,
    )
    model_load_time = time.time() - model_load_start
    logger.info(f"Model loaded in {model_load_time:.1f}s")

    # Start result collector
    collector = ResultCollector(streaming_iterator, total_jobs)
    collector.start()

    logger.info("=" * 60)
    logger.info("INFERENCE STARTED")
    logger.info("=" * 60)
    inference_start = time.time()

    # Wait for all results
    timeout = 1800  # 30 minutes
    while collector.get_count() < total_jobs:
        if shutdown_requested.is_set():
            break
        if time.time() - inference_start > timeout:
            logger.error("Timeout waiting for results!")
            break

        elapsed = int(time.time() - inference_start)
        if elapsed > 0 and elapsed % 60 == 0:
            count = collector.get_count()
            rate = count / elapsed if elapsed > 0 else 0
            logger.info(f"Progress: {count}/{total_jobs} ({rate:.2f} files/sec)")

        time.sleep(1)

    inference_end = time.time()
    inference_time = inference_end - inference_start

    collector.stop()
    pipeline.stop()

    results = collector.get_results()

    # Stats
    successful = [
        r for r in results if not r.get("error") or str(r.get("error")).strip() == ""
    ]
    failed = [
        r for r in results if r.get("error") and str(r.get("error")).strip() != ""
    ]

    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total files: {total_jobs}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info("")
    logger.info("TIMING:")
    logger.info(f"  Model loading: {model_load_time:.1f}s")
    logger.info(f"  Inference time: {inference_time:.1f}s")
    logger.info("")
    logger.info("THROUGHPUT (inference only):")
    if len(successful) > 0 and inference_time > 0:
        throughput = len(successful) / inference_time
        logger.info(f"  {throughput:.2f} files/second")
        logger.info(f"  {inference_time / len(successful):.2f} seconds/file")
    logger.info("=" * 60)

    # Save results to JSON
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path(__file__).parent / "eval_benchmark_results.json"

    output_data = {
        "summary": {
            "total_files": total_jobs,
            "successful": len(successful),
            "failed": len(failed),
            "model_load_time_sec": model_load_time,
            "inference_time_sec": inference_time,
            "throughput_files_per_sec": (
                len(successful) / inference_time if inference_time > 0 else 0
            ),
        },
        "results": [
            {
                "filename": r["filename"],
                "background": r.get("background", []),
                "middle_ground": r.get("middle_ground", []),
                "foreground": r.get("foreground", []),
                "error": r.get("error", ""),
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print sample results
    if successful:
        logger.info("\nSample results:")
        for result in successful[:5]:
            logger.info(f"  {result['filename']}:")
            logger.info(f"    BG: {result.get('background', [])}")
            logger.info(f"    MG: {result.get('middle_ground', [])}")
            logger.info(f"    FG: {result.get('foreground', [])}")

    ray.shutdown()

    return 0 if len(results) >= total_jobs else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark on eval_audio_files")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="eval_audio_files",
        help="Directory with audio files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help="Directory for models",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--distributed-executor-backend",
        type=str,
        default="mp",
        choices=["mp", "ray"],
        help="vLLM distributed executor backend (use 'ray' for TP>1 inside Ray)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Max model context length",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=8,
        help="Max number of sequences for vLLM",
    )
    parser.add_argument(
        "--detector-actors",
        type=int,
        default=1,
        help="Number of detector actors (model replicas)",
    )
    parser.add_argument(
        "--detector-batch",
        type=int,
        default=4,
        help="Detector batch size",
    )
    parser.add_argument(
        "--preprocessor-actors",
        type=int,
        default=4,
        help="Number of preprocessor actors",
    )
    parser.add_argument(
        "--preprocessor-batch",
        type=int,
        default=8,
        help="Preprocessor batch size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    print(
        f"""
Eval Benchmark Configuration:
  Data dir: {args.data_dir}
  Model: {args.model}
  Dtype: {args.dtype}
  Tensor Parallel: {args.tensor_parallel_size}
  Distributed Backend: {args.distributed_executor_backend}
  GPU Memory: {args.gpu_memory_utilization}
  Max Model Len: {args.max_model_len}
  Max Num Seqs: {args.max_num_seqs}
  Detector Actors: {args.detector_actors}
  Detector Batch: {args.detector_batch}
  Preprocessor Actors: {args.preprocessor_actors}
  Preprocessor Batch: {args.preprocessor_batch}
"""
    )

    exit_code = main(
        data_dir=args.data_dir,
        model_name=args.model,
        models_dir=args.models_dir,
        dtype=DTYPE_MAP[args.dtype],
        tensor_parallel_size=args.tensor_parallel_size,
        distributed_executor_backend=args.distributed_executor_backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        detector_actors=args.detector_actors,
        detector_batch_size=args.detector_batch,
        preprocessor_actors=args.preprocessor_actors,
        preprocessor_batch_size=args.preprocessor_batch,
        output_file=args.output,
    )

    import sys

    sys.exit(exit_code)
