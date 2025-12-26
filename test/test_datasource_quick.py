"""Quick test to verify StreamingDatasource works with Ray Data.

Tests with configurable number of jobs and burst probability to simulate
real-time job submission at scale (1k+ jobs).
"""

import ray
import time
import hashlib
import uuid
import random
import threading
import argparse
from pathlib import Path
from ray.util.queue import Queue

from src.streaming_pipeline.streaming_datasource import (
    QueueStreamingDatasource,
    StreamingDatasourceConfig,
    STOP_SENTINEL,
)
from src.pipelines.instrument_detection.data_classes import InstrumentDetectJob


# Audio files directory
AUDIO_DIR = Path("/Users/rajsingh/Desktop/code/music_audio_analyzer/audio_files")


def create_job_from_file(filepath: Path) -> InstrumentDetectJob:
    """Create a job from a file path."""
    audio_bytes = filepath.read_bytes()
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    song_id = f"trk_{uuid.uuid4().hex[:12]}"
    song_hash = hashlib.sha256(audio_bytes).hexdigest()
    created_at = int(time.time())

    audio_ref = ray.put(audio_bytes)

    return InstrumentDetectJob(
        job_id=job_id,
        created_at=created_at,
        song_id=song_id,
        song_hash=song_hash,
        audio_ref=audio_ref,
        filename=filepath.name,
    )


def job_to_row(job: InstrumentDetectJob) -> dict:
    """Convert an InstrumentDetectJob to a row dict for the datasource."""
    return {
        "job_id": job.job_id,
        "song_id": job.song_id,
        "song_hash": job.song_hash,
        "filename": job.filename,
        "audio_ref": job.audio_ref,
    }


class StreamingJobProducer:
    """Produces jobs at random intervals to simulate real-time job submission."""

    def __init__(
        self,
        job_queue: Queue,
        audio_files: list,
        total_jobs: int,
        min_delay_ms: int = 10,
        max_delay_ms: int = 100,
        burst_probability: float = 0.3,
        burst_size_min: int = 5,
        burst_size_max: int = 20,
    ):
        self.job_queue = job_queue
        self.audio_files = audio_files
        self.total_jobs = total_jobs
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.burst_probability = burst_probability
        self.burst_size_min = burst_size_min
        self.burst_size_max = burst_size_max

        self.jobs_submitted = 0
        self.stop_event = threading.Event()
        self.thread = None

    def _submit_job(self) -> bool:
        """Submit a single job. Returns True if submitted."""
        if self.jobs_submitted >= self.total_jobs:
            return False

        audio_file = random.choice(self.audio_files)
        job = create_job_from_file(audio_file)
        row = job_to_row(job)
        self.job_queue.put(row)
        self.jobs_submitted += 1
        return True

    def _producer_loop(self):
        """Main producer loop that runs in a separate thread."""
        print(f"Producer started. Will submit {self.total_jobs} jobs.")

        while not self.stop_event.is_set() and self.jobs_submitted < self.total_jobs:
            # Decide if this is a burst or single submission
            if random.random() < self.burst_probability:
                burst_size = min(
                    random.randint(self.burst_size_min, self.burst_size_max),
                    self.total_jobs - self.jobs_submitted,
                )
                for _ in range(burst_size):
                    if not self._submit_job():
                        break
            else:
                self._submit_job()

            if self.jobs_submitted < self.total_jobs:
                delay_ms = random.randint(self.min_delay_ms, self.max_delay_ms)
                time.sleep(delay_ms / 1000.0)

            if self.jobs_submitted % 100 == 0:
                print(
                    f"  Producer: {self.jobs_submitted}/{self.total_jobs} jobs submitted"
                )

        # Don't send STOP_SENTINEL - use expected_items/max_items instead
        # This avoids race conditions with concurrent producer/consumer
        print(f"Producer finished. Submitted {self.jobs_submitted} jobs")

    def start(self):
        """Start the producer thread."""
        self.thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the producer thread."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def is_done(self) -> bool:
        """Check if all jobs have been submitted."""
        return self.jobs_submitted >= self.total_jobs


def test_datasource_read(
    total_jobs: int = 100,
    min_delay_ms: int = 5,
    max_delay_ms: int = 50,
    burst_probability: float = 0.3,
    burst_size_min: int = 5,
    burst_size_max: int = 20,
    batch_size: int = 10,
):
    """Test that we can read from the datasource with streaming job producer."""
    ray.init(ignore_reinit_error=True)

    # Find audio files
    audio_files = list(AUDIO_DIR.glob("*.mp3")) + list(AUDIO_DIR.glob("*.wav"))
    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}")
        return

    print(f"Found {len(audio_files)} audio files")
    print(f"Will create {total_jobs} jobs with burst_probability={burst_probability}")

    # Create queue
    queue = Queue()

    # Create and start producer
    producer = StreamingJobProducer(
        job_queue=queue,
        audio_files=audio_files,
        total_jobs=total_jobs,
        min_delay_ms=min_delay_ms,
        max_delay_ms=max_delay_ms,
        burst_probability=burst_probability,
        burst_size_min=burst_size_min,
        burst_size_max=burst_size_max,
    )

    # Create datasource with expected_items to handle concurrent producer/consumer
    # No STOP_SENTINEL needed - reader stops after reading expected_items
    config = StreamingDatasourceConfig(
        parallelism=1,
        batch_size=batch_size,
        batch_timeout=1.0,
    )
    datasource = QueueStreamingDatasource(
        queue=queue,
        item_to_row_fn=lambda x: x,
        config=config,
        expected_items=total_jobs,  # Reader will stop after this many items
    )

    print("\nCreating dataset from datasource...")
    try:
        ds = ray.data.read_datasource(datasource)
        print(f"Dataset created: {ds}")

        # Start producer AFTER dataset is created (concurrent mode)
        print("Starting producer (concurrent with consumer)...")
        producer.start()

        # Read batches
        print("\nReading batches:")
        start_time = time.time()
        total_rows = 0
        batch_count = 0

        for batch in ds.iter_batches(batch_size=batch_size):
            batch_count += 1
            rows_in_batch = len(batch.get("filename", []))
            total_rows += rows_in_batch

            # Show batch structure for first batch
            if batch_count == 1:
                print(f"\n  === Batch Structure (type: {type(batch).__name__}) ===")
                print(f"  Keys: {list(batch.keys())}")
                for key, value in batch.items():
                    print(
                        f"    '{key}': {type(value).__name__}, shape/len={getattr(value, 'shape', len(value))}"
                    )

                print(f"\n  === First Row ===")
                for key, value in batch.items():
                    first_val = value[0] if len(value) > 0 else None
                    # Truncate long values for display
                    val_str = str(first_val)
                    if len(val_str) > 80:
                        val_str = val_str[:80] + "..."
                    print(f"    '{key}': {val_str}")
                print()

            # Log every 100 rows
            if total_rows % 100 < rows_in_batch or batch_count <= 3:
                elapsed = time.time() - start_time
                rate = total_rows / elapsed if elapsed > 0 else 0
                print(
                    f"  Consumer: {total_rows} rows read, "
                    f"batch={batch_count}, rate={rate:.1f} rows/sec"
                )

        elapsed = time.time() - start_time
        rate = total_rows / elapsed if elapsed > 0 else 0

        print(f"\n✓ Success!")
        print(f"  Total batches: {batch_count}")
        print(f"  Total rows: {total_rows}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Rate: {rate:.1f} rows/sec")

    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
    finally:
        producer.stop()
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test StreamingDatasource")
    parser.add_argument("--jobs", type=int, default=100, help="Total jobs to create")
    parser.add_argument(
        "--min-delay", type=int, default=5, help="Min delay between jobs (ms)"
    )
    parser.add_argument(
        "--max-delay", type=int, default=50, help="Max delay between jobs (ms)"
    )
    parser.add_argument(
        "--burst-prob", type=float, default=0.3, help="Burst probability"
    )
    parser.add_argument("--burst-min", type=int, default=5, help="Min burst size")
    parser.add_argument("--burst-max", type=int, default=20, help="Max burst size")
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for reading"
    )

    args = parser.parse_args()

    test_datasource_read(
        total_jobs=args.jobs,
        min_delay_ms=args.min_delay,
        max_delay_ms=args.max_delay,
        burst_probability=args.burst_prob,
        burst_size_min=args.burst_min,
        burst_size_max=args.burst_max,
        batch_size=args.batch_size,
    )
