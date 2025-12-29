# Real-time and bulk instrument detection for downstream stem splitting and analysis

As of Christmas 2025:
This repo started out with ideas to help producers when they hear a song, and want to reproduce it or produce something similar. I also wanted to use open source models, for all the benefits they come with - licensing, paying for APIs, etc.

I started off with the idea of advanced stem splitting to be able to feed as downstream data to to some sort of analysis. Currently this exists as a fast streaming pipeline to be able to take thousands of songs (potentially spanning many users). It does instrument detection as of now using Qwen Omni, but more to come! Eventually this will be a fully fleshed out agentic design to do some cool stuff with music, designed to help producers. Design docs can be found in the `docs/` folder.

Once you have the Dockerfile running, the environment should have everything you need to run it (assuming you have set your PYHONPATH at the root).


The batch test script is now tested with an H100 and vLLM. You can build the image with 
```bash
docker buildx build \
    --platform linux/amd64 \
    -f Dockerfile_vllm_h100 \
    -t $(YOUR_IMAGE_TAG) \
    .
```
Then, run vLLM inference with your desired audio files and results
```bash
python src/experiments/run_vllm_cot_test_data.py --data_dir your_audio_files/ --batch_size 4 --results_dir results_dir/
```

Example results:
```python
song_name = "Drake - Teenage Fever"

print(song_name)
for ground in ["background", "middle_ground", "foreground"]:
    print(f"{ground}: {results_indexed.loc[song_name].loc[ground]}")

'''
Drake - Teenage Fever
background: ['sub bass', 'kick drum', 'hi-hats', 'snare']
middle_ground: ['electric piano', 'synth pad', 'sampled loop']
foreground: ['male lead vocal', 'ad-libs', 'spoken word']
'''
```