"""
Agents for the instrument detection pipeline.
"""

from .audio_preprocessor import AudioPreprocessorAgent
from .instrument_detector import InstrumentDetectorAgent, InstrumentDetectorCoTAgent

__all__ = [
    "AudioPreprocessorAgent",
    "InstrumentDetectorAgent",
    "InstrumentDetectorCoTAgent",
]
