"""
Test script to verify waveform ObjectRefs survive PyArrow serialization.

This tests the pipeline data flow WITHOUT GPU/vLLM components:
1. Audio bytes -> Preprocessor -> waveform_ref (ObjectRef)
2. waveform_ref goes through PyArrow serialization between stages
3. Mock detector tries to ray.get() the waveform_ref

Run with: python test/test_waveform_passing.py
"""

import ray
import time
import numpy as np
from pathlib import Path
from loguru import logger
from ray.util.queue import Queue

from src.streaming_pipeline import (
    AgentRayComputeConfig,
    AgentStage,
    QueueStreamingDatasource,
    StreamingDatasourceConfig,
    StreamingPipeline,
)
from src.streaming_pipeline.agent import Agent
from src.pipelines.instrument_detection.agents.audio_preprocessor import (
    AudioPreprocessorAgent,
)


class MockDetectorAgent(Agent):
    """
    Mock detector that just deserializes waveforms and validates them.
    No GPU or model loading required.
    """

    def setup(self):
        logger.info("MockDetectorAgent setup complete")

    def process_batch(self, items):
        import time

        logger.info(f"MockDetector received batch of {len(items)} items")
        results = []
        now = int(time.time())

        for idx, item in enumerate(items):
            filename = item.get("filename", f"unknown_{idx}")
            error_value = item.get("error", "")

            logger.info(
                f"Item {idx} '{filename}':\n"
                f"  keys: {list(item.keys())}\n"
                f"  waveform_bytes type: {type(item.get('waveform_bytes'))}\n"
                f"  waveform_bytes len: {len(item.get('waveform_bytes', b'')) if item.get('waveform_bytes') else 0}\n"
                f"  error: '{error_value}'"
            )

            # Check for preprocessing errors
            # Note: error field should be empty string "" for success, non-empty for error
            has_real_error = (
                error_value is not None
                and str(error_value).strip() != ""
                and str(error_value).strip() != "nan"  # Legacy nan handling
            )

            if has_real_error:
                logger.error(
                    f"Item '{filename}' has preprocessing error: {error_value}"
                )
                results.append(
                    {
                        "job_id": item.get("job_id"),
                        "filename": filename,
                        "success": False,
                        "error": str(error_value),
                        "waveform_shape": None,
                    }
                )
                continue

            # Get waveform_bytes and deserialize
            try:
                waveform_bytes = item.get("waveform_bytes")

                if waveform_bytes is None or len(waveform_bytes) == 0:
                    raise ValueError("waveform_bytes is None or empty")

                # Handle PyArrow binary types - convert to Python bytes if needed
                if hasattr(waveform_bytes, "as_py"):
                    waveform_bytes = waveform_bytes.as_py()
                elif hasattr(waveform_bytes, "tobytes"):
                    waveform_bytes = waveform_bytes.tobytes()

                # Deserialize bytes to numpy array
                waveform = np.frombuffer(waveform_bytes, dtype=np.float32)

                if waveform.size == 0:
                    raise ValueError("Waveform is empty (size=0)")

                logger.info(
                    f"SUCCESS: '{filename}' waveform shape={waveform.shape}, dtype={waveform.dtype}"
                )
                results.append(
                    {
                        "job_id": item.get("job_id"),
                        "filename": filename,
                        "success": True,
                        "error": None,
                        "waveform_shape": list(waveform.shape),
                        "duration_seconds": item.get("duration_seconds"),
                    }
                )

            except Exception as e:
                logger.error(f"FAILED: '{filename}' - {e}")
                results.append(
                    {
                        "job_id": item.get("job_id"),
                        "filename": filename,
                        "success": False,
                        "error": str(e),
                        "waveform_shape": None,
                    }
                )

        return results


def create_test_job(filepath: Path) -> dict:
    """Create a test job from an audio file."""
    import hashlib
    import uuid

    audio_bytes = filepath.read_bytes()
    return {
        "job_id": f"test_{uuid.uuid4().hex[:8]}",
        "song_id": f"song_{uuid.uuid4().hex[:8]}",
        "song_hash": hashlib.sha256(audio_bytes).hexdigest()[:16],
        "filename": filepath.name,
        "audio_bytes": audio_bytes,
    }


def main():
    # Find audio files
    audio_dir = Path(__file__).parent.parent / "audio_files"
    audio_files = list(audio_dir.glob("*.mp3"))[:5]  # Just test with 5 files

    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio files for testing")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    logger.info("Ray initialized")

    # Create job queue
    job_queue = Queue(maxsize=100)

    # Create datasource
    datasource = QueueStreamingDatasource(
        queue=job_queue,
        item_to_row_fn=lambda x: x,
        config=StreamingDatasourceConfig(
            parallelism=1,
            batch_size=4,
            batch_timeout=1.0,
            poll_interval=0.1,
            max_items=len(audio_files),
        ),
    )

    # Create pipeline with preprocessor and mock detector
    pipeline = StreamingPipeline(
        datasource=datasource,
        stages=[
            AgentStage(
                agent=AudioPreprocessorAgent(target_sr=16000),
                config=AgentRayComputeConfig(
                    num_actors=1,
                    batch_size=4,
                    num_cpus=1.0,
                ),
                name="AudioPreprocessor",
            ),
            AgentStage(
                agent=MockDetectorAgent(),
                config=AgentRayComputeConfig(
                    num_actors=1,
                    batch_size=4,
                    num_cpus=1.0,
                ),
                name="MockDetector",
            ),
        ],
        name="TestPipeline",
    )

    # Submit jobs
    logger.info("Submitting test jobs...")
    for audio_file in audio_files:
        job = create_test_job(audio_file)
        job_queue.put(job)
        logger.info(f"Submitted: {job['filename']}")

    # Signal end of jobs
    job_queue.put("__STREAMING_DATASOURCE_STOP_SENTINEL__")

    # Process results
    logger.info("Processing pipeline...")
    results = []
    start_time = time.time()

    try:
        for batch in pipeline.stream(batch_size=1):
            if batch:
                keys = list(batch.keys())
                if keys:
                    n_items = len(batch[keys[0]])
                    for i in range(n_items):
                        result = {k: batch[k][i] for k in keys}
                        results.append(result)

                        if result.get("success"):
                            logger.info(
                                f"RESULT: {result['filename']} -> SUCCESS, "
                                f"shape={result.get('waveform_shape')}"
                            )
                        else:
                            logger.warning(
                                f"RESULT: {result['filename']} -> FAILED: {result.get('error')}"
                            )

            if len(results) >= len(audio_files):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    elapsed = time.time() - start_time
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total files: {len(audio_files)}")
    logger.info(f"Results received: {len(results)}")
    logger.info(f"  - Successful: {len(successful)}")
    logger.info(f"  - Failed: {len(failed)}")
    logger.info(f"Time: {elapsed:.2f}s")

    if failed:
        logger.info("\nFailed items:")
        for r in failed:
            logger.info(f"  - {r['filename']}: {r.get('error')}")

    # Cleanup
    pipeline.stop()
    ray.shutdown()

    # Return exit code
    if len(successful) == len(audio_files):
        logger.info("\nALL TESTS PASSED!")
        return 0
    else:
        logger.error(f"\nTESTS FAILED: {len(failed)}/{len(audio_files)} failed")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
