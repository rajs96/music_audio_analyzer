from torch.utils.data import Dataset
from pathlib import Path
from transformers import Qwen3OmniMoeProcessor
import torchaudio
import numpy as np


class QwenOmniDataset(Dataset):
    def __init__(
        self, data_dir: str, processor: Qwen3OmniMoeProcessor, target_sr: int = 16000
    ):
        # get all files
        self.files = [str(f) for f in Path(data_dir).glob("**/*.mp3")]
        self.processor = processor
        self.target_sr = target_sr

    def __len__(self):
        return len(self.files)

    def decode_audio(self, filepath: str) -> tuple:
        """Decode audio file to waveform at target sample rate."""
        wav, sr = torchaudio.load(filepath)

        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to target_sr
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.target_sr
            )

        return wav.squeeze(0).numpy()

    def __getitem__(self, idx: int | slice):
        filenames = [self.files[idx]] if isinstance(idx, int) else self.files[idx]
        conversations = []

        for filename in filenames:
            waveform = self.decode_audio(filename)

            conversation_filename = []
            conversation_filename.append(self.get_system_prompt())
            conversation_filename.append(self.get_user_prompt(waveform))
            conversations.append(conversation_filename)

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return filenames, inputs

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
