from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoProcessor


class QwenOmniDataset(Dataset):
    def __init__(self, data_dir: str, processor: AutoProcessor):
        # get all files
        self.files = [f for f in Path(data_dir).glob("**/*.mp3")]
        self.processor = processor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int, slice: slice):
        filenames = [self.files[idx] if isinstance(idx, int) else self.files[idx]]
        conversations = []
        for filename in filenames:
            conversation_filename = []
            conversation_filename.append(self.get_system_prompt())
            conversation_filename.append(self.get_user_prompt(filename))
            conversations.append(conversation_filename)

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return inputs

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

    def get_user_prompt(self, filename: str):
        return {
            "role": "user",
            "content": [{"type": "audio", "audio": filename}],
        }
