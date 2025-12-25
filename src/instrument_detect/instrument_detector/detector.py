import time
import traceback
from typing import List, Optional
import numpy as np
import torch
import ray
from ray.util.queue import Queue, Empty
from loguru import logger

from src.instrument_detect.data_classes import PreprocessedAudio, InstrumentDetectResult
from src.instrument_detect.models.load_qwen import load_model_and_processor


class DetectorError(Exception):
    """Raised when detector encounters a fatal error."""

    pass


@ray.remote
class DetectorActor:
    """
    Streaming detector actor that pulls preprocessed audio from input queue,
    runs Qwen inference, and pushes results to output queue.

    Resource configuration is done at actor creation time via .options():
        DetectorActor.options(num_gpus=1, max_concurrency=1).remote(...)
    """

    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        batch_size: int = 4,
        max_wait_ms: int = 100,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.processed_count = 0
        self._error: Optional[str] = None
        self._stopped = False

        # Timing metrics
        self.total_inference_time_ms = 0.0
        self.batch_count = 0

        # Load model on init
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading model {model_name} on {self.device}")
            self.model, self.processor = load_model_and_processor(
                model_name, self.device
            )
            self.model.eval()
            logger.info("Model loaded")
        except Exception as e:
            error_msg = f"Failed to load model: {type(e).__name__}: {e}"
            logger.error(error_msg)
            self._error = error_msg
            raise DetectorError(error_msg) from e

    def is_healthy(self) -> bool:
        """Check if actor is healthy."""
        return self._error is None and not self._stopped

    def get_error(self) -> Optional[str]:
        """Get the error message if actor failed."""
        return self._error

    def stop(self):
        """Signal actor to stop."""
        logger.info("DetectorActor received stop signal")
        self._stopped = True

    def get_system_prompt(self):
        text = """
        You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.
        You are also an expert in detecting what instruments are being played in a song.

        You will be given a song and you will need to detect what instruments are being played in the song.
        Return a list of strings, each string is the name of an instrument.

        Only use the following strings:
        - drums
        - bass
        - electric_guitar
        - acoustic_guitar
        - piano
        - synthesizer
        - strings
        - wind
        - vocals

        Example output 1: ['drums', 'electric_guitar', 'piano', 'vocals']
        Example output 2: ['acoustic_guitar', 'piano', 'vocals', 'bass']
        Example output 3: ['piano', 'vocals']
        """
        return {
            "role": "system",
            "content": [{"type": "text", "text": text}],
        }

    def get_user_prompt(self, waveform: np.ndarray):
        return {
            "role": "user",
            "content": [{"type": "audio", "audio": waveform}],
        }

    def tokenize(self, waveforms: List[np.ndarray]):
        """Convert waveforms to model inputs."""
        conversations = []
        for waveform in waveforms:
            conversation = [self.get_system_prompt(), self.get_user_prompt(waveform)]
            conversations.append(conversation)

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return inputs

    def _pull_batch(self, max_items: int) -> List[PreprocessedAudio]:
        """Pull up to max_items from queue without blocking."""
        items = []
        for _ in range(max_items):
            try:
                item = self.input_queue.get_nowait()
                items.append(item)
            except Empty:
                break
        return items

    def predict_batch(self, batch: List[PreprocessedAudio]) -> List[str]:
        """Run inference on a batch of preprocessed audio."""
        waveforms = [item.waveform for item in batch]

        # Tokenize
        inputs = self.tokenize(waveforms)
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # Decode
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return responses

    def run_forever(self):
        """Continuously pull from input queue, run inference, and push results."""
        logger.info("DetectorActor started")

        try:
            while not self._stopped:
                # Pull initial batch
                batch = self._pull_batch(self.batch_size)
                if not batch:
                    time.sleep(0.01)
                    continue

                # Try to top up batch for better GPU utilization
                deadline = time.time() + (self.max_wait_ms / 1000.0)
                while (
                    len(batch) < self.batch_size
                    and time.time() < deadline
                    and not self._stopped
                ):
                    remaining = self.batch_size - len(batch)
                    extra = self._pull_batch(remaining)
                    if not extra:
                        time.sleep(0.005)
                        continue
                    batch.extend(extra)

                if self._stopped:
                    logger.info("DetectorActor stopping, discarding batch")
                    break

                # Run inference with timing
                logger.info(f"Running inference on batch of {len(batch)}")
                inference_start = time.time()
                predictions = self.predict_batch(batch)
                inference_time_ms = (time.time() - inference_start) * 1000

                # Update timing metrics
                self.total_inference_time_ms += inference_time_ms
                self.batch_count += 1
                self.processed_count += len(batch)

                avg_batch_time = self.total_inference_time_ms / self.batch_count
                avg_per_example = self.total_inference_time_ms / self.processed_count
                logger.info(
                    f"Batch inference: {inference_time_ms:.1f}ms ({inference_time_ms/len(batch):.1f}ms/example) | "
                    f"Avg: {avg_batch_time:.1f}ms/batch, {avg_per_example:.1f}ms/example"
                )

                # Create results and push to output queue
                now = int(time.time())
                for item, prediction in zip(batch, predictions):
                    result = InstrumentDetectResult(
                        job_id=item.job_id,
                        song_id=item.song_id,
                        song_hash=item.song_hash,
                        filename=item.filename,
                        instruments=self._parse_instruments(prediction),
                        detected_at=now,
                    )
                    self.output_queue.put(result)
                    logger.info(
                        f"Detected instruments for {item.filename}: {result.instruments}"
                    )

        except Exception as e:
            error_msg = f"DetectorActor fatal error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self._error = error_msg
            raise DetectorError(error_msg) from e
        finally:
            logger.info(
                f"DetectorActor stopped. Processed {self.processed_count} items."
            )

    def _parse_instruments(self, prediction: str) -> List[str]:
        """Parse the model output to extract instrument list."""
        # Try to extract list from response
        # Model should return something like: ['drums', 'guitar', 'vocals']
        try:
            # Simple parsing - look for list pattern
            import ast

            start = prediction.find("[")
            end = prediction.rfind("]") + 1
            if start != -1 and end > start:
                return ast.literal_eval(prediction[start:end])
        except:
            pass
        return [prediction.strip()]

    def get_stats(self) -> dict:
        avg_batch_time_ms = (
            self.total_inference_time_ms / self.batch_count
            if self.batch_count > 0
            else 0.0
        )
        avg_per_example_ms = (
            self.total_inference_time_ms / self.processed_count
            if self.processed_count > 0
            else 0.0
        )
        return {
            "processed_count": self.processed_count,
            "batch_count": self.batch_count,
            "total_inference_time_ms": self.total_inference_time_ms,
            "avg_batch_time_ms": avg_batch_time_ms,
            "avg_per_example_ms": avg_per_example_ms,
        }
