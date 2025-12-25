import signal
import sys
import time
import traceback
import ray
import torch
from ray.util.queue import Queue, Empty
from loguru import logger
from typing import Optional, List, Callable

from src.instrument_detect.data_classes import (
    InstrumentDetectJob,
    InstrumentDetectResult,
)
from src.instrument_detect.instrument_detector.preprocessor import (
    PreprocessorActor,
    PreprocessorError,
)
from src.instrument_detect.instrument_detector.detector import DetectorActor


class DispatcherError(Exception):
    """Raised when dispatcher encounters a fatal error."""

    pass


@ray.remote
class PreprocessorDispatcher:
    """
    Pulls jobs from input queue, submits to preprocessor pool,
    and pushes results to waveform queue.

    Applies backpressure using ray.wait to limit pending tasks.
    """

    def __init__(
        self,
        job_queue: Queue,
        waveform_queue: Queue,
        pool_size: int = 4,
        batch_size: int = 8,
        max_wait_ms: int = 50,
        max_pending_tasks: int = 16,
        max_waveform_queue_size: int = 50,
        preprocessor_num_cpus: float = 1.0,
        preprocessor_max_concurrency: int = 1,
    ):
        self.job_queue = job_queue
        self.waveform_queue = waveform_queue
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.max_pending_tasks = max_pending_tasks
        self.max_waveform_queue_size = max_waveform_queue_size
        self.dispatched_count = 0
        self._error: Optional[str] = None
        self._stopped = False
        self._failed_jobs: List[str] = []  # Track failed job IDs

        # Create preprocessor actors with resource configuration
        self.pool_size = pool_size
        self.preprocessor_num_cpus = preprocessor_num_cpus
        self.preprocessor_max_concurrency = preprocessor_max_concurrency

        self.preprocessors = [
            PreprocessorActor.options(
                num_cpus=preprocessor_num_cpus,
                max_concurrency=preprocessor_max_concurrency,
            ).remote()
            for _ in range(pool_size)
        ]
        self.pending_tasks: List[ray.ObjectRef] = []
        self.task_to_actor: dict = {}  # Map task ref to actor for round-robin
        self.next_actor_idx = 0

        logger.info(
            f"Created {pool_size} preprocessor actors "
            f"(num_cpus={preprocessor_num_cpus}, max_concurrency={preprocessor_max_concurrency})"
        )

    def is_healthy(self) -> bool:
        """Check if dispatcher is healthy."""
        return self._error is None and not self._stopped

    def get_error(self) -> Optional[str]:
        """Get the error message if dispatcher failed."""
        return self._error

    def stop(self):
        """Signal dispatcher and all preprocessors to stop."""
        logger.info("PreprocessorDispatcher received stop signal")
        self._stopped = True
        # Stop all preprocessor actors
        for preprocessor in self.preprocessors:
            try:
                ray.get(preprocessor.stop.remote(), timeout=1.0)
            except Exception:
                pass  # Actor may already be dead

    def _pull_batch(self, max_items: int) -> List[InstrumentDetectJob]:
        """Pull up to max_items from queue without blocking."""
        items = []
        for _ in range(max_items):
            try:
                item = self.job_queue.get_nowait()
                items.append(item)
            except Empty:
                break
        return items

    def _get_next_actor(self):
        """Round-robin actor selection."""
        actor = self.preprocessors[self.next_actor_idx]
        self.next_actor_idx = (self.next_actor_idx + 1) % self.pool_size
        return actor

    def _collect_completed(self, block: bool = False) -> int:
        """Collect completed tasks and push to waveform queue. Returns count collected."""
        if not self.pending_tasks:
            return 0

        if block:
            # Wait for at least one to complete (with timeout to check stop signal)
            ready, self.pending_tasks = ray.wait(
                self.pending_tasks, num_returns=1, timeout=1.0
            )
        else:
            # Non-blocking check
            ready, self.pending_tasks = ray.wait(
                self.pending_tasks, num_returns=len(self.pending_tasks), timeout=0
            )

        collected = 0
        for ref in ready:
            try:
                preprocessed = ray.get(ref)
                self.waveform_queue.put(preprocessed)
                self.dispatched_count += 1
                collected += 1
            except PreprocessorError as e:
                # Preprocessor had a fatal error - track and continue
                logger.error(f"Preprocessor error: {e}")
                self._failed_jobs.append(str(e))
            except ray.exceptions.RayActorError as e:
                # Actor died - this is serious
                error_msg = f"Preprocessor actor died: {e}"
                logger.error(error_msg)
                self._error = error_msg
                raise DispatcherError(error_msg) from e
            except Exception as e:
                logger.error(
                    f"Error getting preprocessed result: {type(e).__name__}: {e}"
                )
                self._failed_jobs.append(str(e))

        return collected

    def _apply_backpressure(self):
        """Block until we're under the limits. Returns False if stopped."""
        # Backpressure on pending tasks
        while len(self.pending_tasks) >= self.max_pending_tasks and not self._stopped:
            logger.debug(
                f"Backpressure: {len(self.pending_tasks)} pending tasks, waiting..."
            )
            self._collect_completed(block=True)

        # Backpressure on waveform queue
        while (
            self.waveform_queue.qsize() >= self.max_waveform_queue_size
            and not self._stopped
        ):
            logger.debug(
                f"Backpressure: waveform queue size {self.waveform_queue.qsize()}, waiting..."
            )
            time.sleep(0.01)

        return not self._stopped

    def run_forever(self):
        """Continuously dispatch jobs to preprocessor pool with backpressure."""
        logger.info("Dispatcher started")

        try:
            while not self._stopped:
                # Collect any completed tasks (non-blocking)
                self._collect_completed(block=False)

                # Apply backpressure before pulling more jobs
                if not self._apply_backpressure():
                    break

                # Pull batch from job queue
                batch = self._pull_batch(self.batch_size)
                if not batch:
                    # No jobs available, try to collect completed tasks
                    if self.pending_tasks:
                        self._collect_completed(block=True)
                    else:
                        time.sleep(0.01)
                    continue

                # Try to top up batch
                deadline = time.time() + (self.max_wait_ms / 1000.0)
                while (
                    len(batch) < self.batch_size
                    and time.time() < deadline
                    and not self._stopped
                ):
                    extra = self._pull_batch(self.batch_size - len(batch))
                    if extra:
                        batch.extend(extra)
                    else:
                        time.sleep(0.005)

                if self._stopped:
                    logger.info("Dispatcher stopping, discarding batch")
                    break

                # Submit jobs to preprocessors (round-robin)
                logger.debug(f"Submitting {len(batch)} jobs to preprocessors")
                for job in batch:
                    actor = self._get_next_actor()
                    task_ref = actor.process.remote(job)
                    self.pending_tasks.append(task_ref)

        except Exception as e:
            error_msg = f"Dispatcher fatal error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self._error = error_msg
            raise DispatcherError(error_msg) from e
        finally:
            logger.info(
                f"Dispatcher stopped. Dispatched {self.dispatched_count} items, {len(self._failed_jobs)} failures."
            )

    def get_stats(self) -> dict:
        return {
            "dispatched_count": self.dispatched_count,
            "pending_tasks": len(self.pending_tasks),
        }


class InstrumentDetectPipeline:
    """
    Orchestrates the streaming pipeline for instrument detection.

    Architecture:
        JobQueue → Dispatcher → [PreprocessorPool] → WaveformQueue → DetectorActor → ResultQueue
    """

    def __init__(
        self,
        job_queue: Queue,
        pool_size: int = 4,
        dispatcher_batch_size: int = 8,
        detector_batch_size: int = 4,
        max_pending_tasks: int = 16,
        max_waveform_queue_size: int = 50,
        result_queue_size: int = 100,
        model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        # Preprocessor resource configuration
        preprocessor_num_cpus: float = 1.0,
        preprocessor_max_concurrency: int = 1,
        # Detector resource configuration
        detector_num_gpus: float = 1.0,
        detector_max_concurrency: int = 1,
        num_detector_actors: int = 1,
        # Model dtype
        dtype: torch.dtype = torch.float32,
    ):
        self.job_queue = job_queue
        self.pool_size = pool_size
        self.dispatcher_batch_size = dispatcher_batch_size
        self.detector_batch_size = detector_batch_size
        self.max_pending_tasks = max_pending_tasks
        self.max_waveform_queue_size = max_waveform_queue_size
        self.model_name = model_name
        self.preprocessor_num_cpus = preprocessor_num_cpus
        self.preprocessor_max_concurrency = preprocessor_max_concurrency
        self.detector_num_gpus = detector_num_gpus
        self.detector_max_concurrency = detector_max_concurrency
        self.num_detector_actors = num_detector_actors
        self.dtype = dtype

        # Create intermediate queues
        self.waveform_queue = Queue(maxsize=max_waveform_queue_size)
        self.result_queue = Queue(maxsize=result_queue_size)

        # Actor references (created on start)
        self.dispatcher = None
        self.detectors: List[ray.actor.ActorHandle] = []
        self._dispatcher_task: Optional[ray.ObjectRef] = None
        self._detector_tasks: List[ray.ObjectRef] = []
        self._shutdown_requested = False

        logger.info(
            f"Pipeline initialized: pool_size={pool_size}, "
            f"preprocessor_num_cpus={preprocessor_num_cpus}, preprocessor_max_concurrency={preprocessor_max_concurrency}, "
            f"detector_num_gpus={detector_num_gpus}, detector_max_concurrency={detector_max_concurrency}, "
            f"num_detector_actors={num_detector_actors}, "
            f"max_pending={max_pending_tasks}, max_waveform_queue={max_waveform_queue_size}"
        )

    def start(self):
        """Start all pipeline actors."""
        logger.info("Starting pipeline...")

        # Start dispatcher (manages preprocessor pool with backpressure)
        self.dispatcher = PreprocessorDispatcher.remote(
            job_queue=self.job_queue,
            waveform_queue=self.waveform_queue,
            pool_size=self.pool_size,
            batch_size=self.dispatcher_batch_size,
            max_pending_tasks=self.max_pending_tasks,
            max_waveform_queue_size=self.max_waveform_queue_size,
            preprocessor_num_cpus=self.preprocessor_num_cpus,
            preprocessor_max_concurrency=self.preprocessor_max_concurrency,
        )
        self._dispatcher_task = self.dispatcher.run_forever.remote()
        logger.info("Started Dispatcher with backpressure")

        # Start detector actors (GPU workers)
        for _ in range(self.num_detector_actors):
            detector = DetectorActor.options(
                num_gpus=self.detector_num_gpus,
                max_concurrency=self.detector_max_concurrency,
            ).remote(
                input_queue=self.waveform_queue,
                output_queue=self.result_queue,
                model_name=self.model_name,
                batch_size=self.detector_batch_size,
                dtype=self.dtype,
            )
            self.detectors.append(detector)
            self._detector_tasks.append(detector.run_forever.remote())

        logger.info(
            f"Started {self.num_detector_actors} DetectorActor(s) "
            f"(num_gpus={self.detector_num_gpus}, max_concurrency={self.detector_max_concurrency})"
        )

        logger.info("Pipeline started!")

    def get_stats(self) -> dict:
        """Get statistics from all actors."""
        stats = {
            "job_queue_size": self.job_queue.qsize(),
            "waveform_queue_size": self.waveform_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
        }

        if self.dispatcher:
            try:
                stats["dispatcher"] = ray.get(self.dispatcher.get_stats.remote())
            except:
                stats["dispatcher"] = {"error": "unavailable"}

        if self.detectors:
            stats["detectors"] = []
            for detector in self.detectors:
                try:
                    stats["detectors"].append(ray.get(detector.get_stats.remote()))
                except:
                    stats["detectors"].append({"error": "unavailable"})

        return stats

    def check_health(self) -> dict:
        """
        Check health of all actors.

        Returns dict with status of each component:
            - 'healthy': bool - True if all actors are healthy
            - 'dispatcher': dict with 'alive' and 'error' keys
            - 'detectors': list of dicts with 'alive' and 'error' keys
        """
        result = {"healthy": True, "dispatcher": {}, "detectors": []}

        # Check dispatcher
        if self.dispatcher:
            try:
                is_healthy = ray.get(self.dispatcher.is_healthy.remote(), timeout=2.0)
                error = ray.get(self.dispatcher.get_error.remote(), timeout=2.0)
                result["dispatcher"] = {
                    "alive": True,
                    "healthy": is_healthy,
                    "error": error,
                }
                if not is_healthy:
                    result["healthy"] = False
            except Exception as e:
                result["dispatcher"] = {"alive": False, "error": str(e)}
                result["healthy"] = False
        else:
            result["dispatcher"] = {"alive": False, "error": "Not started"}
            result["healthy"] = False

        # Check detectors
        if self.detectors:
            for detector in self.detectors:
                try:
                    is_healthy = ray.get(detector.is_healthy.remote(), timeout=2.0)
                    error = ray.get(detector.get_error.remote(), timeout=2.0)
                    result["detectors"].append(
                        {"alive": True, "healthy": is_healthy, "error": error}
                    )
                    if not is_healthy:
                        result["healthy"] = False
                except Exception as e:
                    result["detectors"].append({"alive": False, "error": str(e)})
                    result["healthy"] = False
        else:
            result["detectors"].append({"alive": False, "error": "Not started"})
            result["healthy"] = False

        return result

    def check_for_failures(self, timeout: float = 0) -> Optional[str]:
        """
        Check if any actor tasks have failed.

        Args:
            timeout: How long to wait checking for failures (0 = non-blocking)

        Returns:
            Error message if a failure is detected, None otherwise.
        """
        tasks = []
        if self._dispatcher_task:
            tasks.append(("dispatcher", self._dispatcher_task))
        for i, task in enumerate(self._detector_tasks):
            tasks.append((f"detector_{i}", task))

        if not tasks:
            return None

        # Check if any tasks have completed (which would indicate failure since they run forever)
        refs = [t[1] for t in tasks]
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=timeout)

        for ref in ready:
            name = next(n for n, r in tasks if r == ref)
            try:
                # If we get here, the task finished - which means it failed
                ray.get(ref)
                return f"{name} unexpectedly stopped"
            except Exception as e:
                return f"{name} failed: {type(e).__name__}: {e}"

        return None

    def shutdown(self, timeout: float = 10.0):
        """
        Gracefully shutdown all pipeline actors.

        Args:
            timeout: Max seconds to wait for actors to stop
        """
        if self._shutdown_requested:
            logger.warning("Shutdown already requested")
            return

        self._shutdown_requested = True
        logger.info("Shutting down pipeline...")

        # Signal actors to stop
        stop_tasks = []
        if self.dispatcher:
            try:
                stop_tasks.append(("dispatcher", self.dispatcher.stop.remote()))
            except Exception as e:
                logger.warning(f"Error signaling dispatcher to stop: {e}")

        for i, detector in enumerate(self.detectors):
            try:
                stop_tasks.append((f"detector_{i}", detector.stop.remote()))
            except Exception as e:
                logger.warning(f"Error signaling detector_{i} to stop: {e}")

        # Wait for stop signals to be processed
        for name, ref in stop_tasks:
            try:
                ray.get(ref, timeout=2.0)
                logger.info(f"{name} acknowledged stop signal")
            except Exception as e:
                logger.warning(f"{name} did not acknowledge stop: {e}")

        # Wait for run_forever tasks to complete
        run_tasks = []
        if self._dispatcher_task:
            run_tasks.append(("dispatcher", self._dispatcher_task, self.dispatcher))
        for i, (task, detector) in enumerate(zip(self._detector_tasks, self.detectors)):
            run_tasks.append((f"detector_{i}", task, detector))

        if run_tasks:
            refs = [t[1] for t in run_tasks]
            ready, not_ready = ray.wait(refs, num_returns=len(refs), timeout=timeout)

            for ref in ready:
                name = next(n for n, r, _ in run_tasks if r == ref)
                try:
                    ray.get(ref)
                    logger.info(f"{name} stopped cleanly")
                except Exception as e:
                    logger.warning(f"{name} stopped with error: {e}")

            # Force kill any actors that didn't stop
            for ref in not_ready:
                name, _, actor = next((n, r, a) for n, r, a in run_tasks if r == ref)
                logger.warning(f"{name} did not stop in time, killing...")
                try:
                    ray.kill(actor)
                except Exception:
                    pass

        logger.info("Pipeline shutdown complete")

    def wait_for_failure(self, check_interval: float = 1.0) -> str:
        """
        Block until an actor fails or shutdown is requested.

        Args:
            check_interval: How often to check for failures in seconds

        Returns:
            Error message describing the failure
        """
        while not self._shutdown_requested:
            error = self.check_for_failures(timeout=check_interval)
            if error:
                return error
        return "Shutdown requested"

    def run(
        self,
        on_result: Optional[Callable[[InstrumentDetectResult], None]] = None,
        check_interval: float = 1.0,
    ):
        """
        Run the pipeline with signal handling and auto-shutdown on failure.

        This method:
        - Starts the pipeline
        - Sets up SIGINT/SIGTERM handlers for graceful shutdown
        - Monitors actors for failures
        - Auto-shuts down if any actor fails
        - Optionally processes results via callback

        Args:
            on_result: Optional callback to process results as they arrive
            check_interval: How often to check for actor failures (seconds)

        Returns:
            None on clean shutdown, raises exception on failure
        """
        # Track original signal handlers
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def signal_handler(signum, _frame):
            sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            logger.info(f"Received {sig_name}, initiating shutdown...")
            self._shutdown_requested = True

        try:
            # Install signal handlers
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            logger.info("Signal handlers installed (Ctrl+C to shutdown)")

            # Start the pipeline
            self.start()

            # Main loop: process results and check for failures
            while not self._shutdown_requested:
                # Check for actor failures
                error = self.check_for_failures(timeout=0)
                if error:
                    logger.error(f"Actor failure detected: {error}")
                    self.shutdown()
                    raise RuntimeError(f"Pipeline failed: {error}")

                # Process results if callback provided
                if on_result:
                    try:
                        result = self.result_queue.get(
                            block=True, timeout=check_interval
                        )
                        on_result(result)
                    except Empty:
                        pass  # No results available, continue monitoring
                else:
                    # Just sleep and monitor
                    time.sleep(check_interval)

            # Clean shutdown requested
            logger.info("Shutdown requested, stopping pipeline...")
            self.shutdown()

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.shutdown()
            raise
        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            logger.info("Signal handlers restored")

    def run_async(self, check_interval: float = 1.0):
        """
        Start the pipeline with a background monitor thread that auto-shuts down on failure.

        Use this when you want to interact with the pipeline programmatically
        rather than blocking in run().

        Args:
            check_interval: How often to check for actor failures (seconds)

        Returns:
            self for method chaining
        """
        import threading

        def monitor():
            while not self._shutdown_requested:
                error = self.check_for_failures(timeout=check_interval)
                if error:
                    logger.error(f"Actor failure detected: {error}")
                    self.shutdown()
                    return
            logger.info("Monitor thread exiting")

        # Install signal handlers
        def signal_handler(signum, _frame):
            sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            logger.info(f"Received {sig_name}, initiating shutdown...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Signal handlers installed (Ctrl+C to shutdown)")

        # Start the pipeline
        self.start()

        # Start monitor thread
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
        logger.info("Background monitor started")

        return self


def create_pipeline(
    job_queue: Queue,
    pool_size: int = 4,
    detector_batch_size: int = 4,
) -> InstrumentDetectPipeline:
    """
    Create and start an instrument detection pipeline.

    Args:
        job_queue: Queue containing InstrumentDetectJob items
        pool_size: Number of preprocessor actors in pool
        detector_batch_size: Batch size for GPU inference

    Returns:
        Running InstrumentDetectPipeline
    """
    pipeline = InstrumentDetectPipeline(
        job_queue=job_queue,
        pool_size=pool_size,
        detector_batch_size=detector_batch_size,
    )
    pipeline.start()
    return pipeline
