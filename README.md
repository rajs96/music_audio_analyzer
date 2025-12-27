# Real-time and bulk instrument detection for downstream stem splitting and analysis

As of Christmas 2025:
This repo started out with ideas to help producers when they hear a song, and want to reproduce it or produce something similar. I also wanted to use open source models, for all the benefits they come with - licensing, paying for APIs, etc.

I started off with the idea of advanced stem splitting to be able to feed as downstream data to to some sort of analysis. Currently this exists as a fast streaming pipeline to be able to take thousands of songs (potentially spanning many users). It does instrument detection as of now using Qwen Omni, but more to come! Eventually this will be a fully fleshed out agentic design to do some cool stuff with music, designed to help producers. Design docs can be found in the `docs/` folder.

You can run basic batched inference for instrument detection with the script `src/experiments/run_qwen_test_data.py`

These scripst are tested in NVIDIA A100s, CUDA 12.4, and PyTorch 2.5.1. You can build the Dockerfile with:

```bash
make build_amd_runpods
```

Once you have the Dockerfile running, the environment should have everything you need to run it (assuming you have set your PYHONPATH at the root).


