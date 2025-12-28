import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.load_qwen import load_model_and_processor


class QwenLLMJudge:
    def __init__(self, model_name: str, dtype: torch.dtype, device: str):
        self.model, self.processor = load_model_and_processor(model_name, dtype, device)
        self.model.eval()

    def judge(self, prompt: str, response: str) -> float:
        return self.model.generate(prompt, response)

    def planning_prompt(self):
        pass
