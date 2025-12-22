from typing import List
import ray
from src.instrument_detect.instrument_detector import InstrumentDetector


@ray.remote
class InstrumentDetectWorker:
    def __init__(
        self,
        instrument_detector: InstrumentDetector,
    ):
        """
        Initialize the InstrumentDetectWorker.

        Args:
            instrument_detector: An instance of InstrumentDetector with `process` and `predict`
                methods that take a list of audio bytes (List[bytes]) of arbitrary size and
                return a list of strings (List[str]) representing detected instruments.
        """
        self.instrument_detector = instrument_detector

    def process_batch(self, audio_bytes_list: List[bytes]) -> List[str]:
        if not audio_bytes_list:
            return []

        # Call the instrument detector's predict method
        predictions = self.instrument_detector.predict(audio_bytes_list)
        return predictions
