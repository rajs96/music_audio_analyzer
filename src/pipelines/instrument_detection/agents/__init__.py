"""
Agents for the instrument detection pipeline.
"""

from .audio_preprocessor import AudioPreprocessorAgent
from .instrument_detector import InstrumentDetectorAgent

__all__ = [
    "AudioPreprocessorAgent",
    "InstrumentDetectorAgent",
]
