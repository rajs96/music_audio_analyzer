from torch.utils.data import Dataset
from pathlib import Path
from transformers import Qwen3OmniMoeProcessor
import torchaudio
import numpy as np
from typing import Dict, Any, List


class QwenOmniDataset(Dataset):
    """Base dataset for Qwen audio processing."""

    SYSTEM_PROMPT = """
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
    - lead_vocals
    - backing_vocals

    Example output 1: ['drums', 'electric_guitar', 'piano', 'vocals']
    Example output 2: ['acoustic_guitar', 'piano', 'vocals', 'bass']
    Example output 3: ['piano', 'vocals']
    """

    def __init__(
        self, data_dir: str, processor: Qwen3OmniMoeProcessor, target_sr: int = 16000
    ):
        self.files = [str(f) for f in Path(data_dir).glob("**/*.mp3")]
        self.processor = processor
        self.target_sr = target_sr

    def __len__(self):
        return len(self.files)

    @staticmethod
    def decode_audio(filepath: str, target_sr: int = 16000) -> np.ndarray:
        """Decode audio file to waveform at target sample rate."""
        wav, sr = torchaudio.load(filepath)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)

        return wav.squeeze(0).numpy()

    # ==================== Conversation Builders ====================

    @staticmethod
    def get_system_message(text: str) -> Dict[str, Any]:
        """Build a system message."""
        return {
            "role": "system",
            "content": [{"type": "text", "text": text}],
        }

    @staticmethod
    def get_user_message(
        waveform: np.ndarray = None, text: str = None
    ) -> Dict[str, Any]:
        """Build a user message with optional audio and/or text."""
        content = []
        if waveform is not None:
            content.append({"type": "audio", "audio": waveform})
        if text is not None:
            content.append({"type": "text", "text": text})
        return {
            "role": "user",
            "content": content,
        }

    @classmethod
    def get_conversation(
        cls, waveform: np.ndarray, user_text: str = None
    ) -> List[Dict[str, Any]]:
        """Build a complete conversation for step 1 inference."""
        return [
            cls.get_system_message(cls.SYSTEM_PROMPT),
            cls.get_user_message(waveform, user_text),
        ]

    # ==================== Dataset Methods ====================

    def __getitem__(self, idx: int | slice):
        filenames = [self.files[idx]] if isinstance(idx, int) else self.files[idx]
        waveforms = []
        conversations = []

        for filename in filenames:
            waveform = self.decode_audio(filename, self.target_sr)
            waveforms.append(waveform)
            conversations.append(self.get_conversation(waveform))

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return filenames, waveforms, inputs


class QwenOmniCoTDataset(QwenOmniDataset):
    """Dataset that uses chain-of-thought prompts for instrument detection."""

    SYSTEM_PROMPT = """You are an expert audio analyst and music producer.
    You understand audio in terms of frequency content, rhythm, and mix placement."""

    STEP_1_PROMPT = """Listen to this audio carefully and describe what you hear in each layer of the mix.

    You MUST respond in EXACTLY this format with these three sections:

    **Background (low-end, rhythm):**
    [Describe the bass, kick drum, and rhythmic foundation you actually hear]

    **Middle-ground (harmonic, textural):**
    [Describe the chords, pads, and harmonic/textural elements you actually hear]

    **Foreground (melodic, prominent):**
    [Describe the lead vocals, solos, and main melodies you actually hear]

    Example response:

    **Background (low-end, rhythm):**
    Deep sub bass with a steady pulse, punchy kick drum on the downbeats, crisp hi-hats with a syncopated pattern.

    **Middle-ground (harmonic, textural):**
    Warm electric piano chords, subtle string pad providing harmonic fill, rhythmic acoustic guitar strumming.

    **Foreground (melodic, prominent):**
    Soulful male lead vocal with vibrato, occasional brass stabs accenting the melody.

    IMPORTANT:
    - Only describe sounds you actually hear in the audio
    - Do not write a general description of the track
    - Do not classify the genre
    - Use the exact format above with the three bold headers"""

    STEP_2_PROMPT_TEMPLATE = """Based on your analysis:

    {step_1_response}

    Now return a JSON object with the instruments in each layer. Use descriptive names for the instruments based on what you heard.

    Example format:
    {{
      "background": ["808 kick", "sub bass", "hi-hats"],
      "middle_ground": ["warm synth pad", "electric guitar chords"],
      "foreground": ["female lead vocal", "synth lead"]
    }}

    Return only the JSON:"""

    @classmethod
    def get_conversation(
        cls, waveform: np.ndarray, user_text: str = None
    ) -> List[Dict[str, Any]]:
        """Build step 1 conversation: system has SYSTEM_PROMPT + STEP_1_PROMPT, user has audio."""
        full_system = cls.SYSTEM_PROMPT + "\n\n" + cls.STEP_1_PROMPT
        return [
            cls.get_system_message(full_system),
            cls.get_user_message(waveform, user_text),
        ]

    @classmethod
    def get_step_2_conversation(
        cls, waveform: np.ndarray, step_1_response: str
    ) -> List[Dict[str, Any]]:
        """Build step 2 conversation: includes audio + step 1 prompt + step 2 template."""
        step_2_text = (
            cls.STEP_1_PROMPT
            + "\n\n"
            + cls.STEP_2_PROMPT_TEMPLATE.format(step_1_response=step_1_response)
        )
        return [
            cls.get_system_message(cls.SYSTEM_PROMPT),
            cls.get_user_message(waveform, step_2_text),
        ]
