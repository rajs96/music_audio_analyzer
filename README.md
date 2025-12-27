# Real-time and bulk instrument detection for downstream stem splitting and analysis

As of Christmas 2025:
This repo started out with ideas to help producers when they hear a song, and want to reproduce it or produce something similar. I also wanted to use open source models, for all the benefits they come with - licensing, paying for APIs, etc.

I started off with the idea of advanced stem splitting to be able to feed as downstream data to to some sort of analysis. Currently this exists as a fast streaming pipeline to be able to take thousands of songs (potentially spanning many users). It does instrument detection as of now using Qwen Omni, but more to come! Eventually this will be a fully fleshed out agentic design to do some cool stuff with music, designed to help producers. Design docs can be found in the `docs/` folder.

You can run basic batched inference for instrument detection as so:

```bash
(your-venv) pip install torchaudio torchvision
(your-venv) pip install -r src/experiments/requirements.txt
(your-venv) python src/experiments/run_qwen_test_data.py
```


