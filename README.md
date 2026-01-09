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

## Streaming Pipeline

The streaming pipeline allows you to continuously submit audio files for instrument detection while the GPU is actively processing. Jobs can be added at any time - the pipeline handles late arrivals seamlessly.

### Running the Late Job Streaming Test

This test demonstrates adding batches of jobs at intervals while the GPU processes:

```bash
python test/test_streaming_late_jobs.py --batch1 5 --batch2 5 --batch3 5 --delay 30
```

Parameters:
- `--batch1`: First batch of jobs (submitted immediately)
- `--batch2`: Second batch (submitted after delay while GPU is processing)
- `--batch3`: Third batch (submitted after another delay)
- `--delay`: Seconds between batch submissions (default: 120)

### Example Output

```
LATE JOB TEST RESULTS
============================================================
Total jobs submitted: 15
Total results received: 15
  - Successful: 15
  - Failed: 0
Total time: 101.62 seconds (includes model loading)

Sample successful results:
  - Matt Quentin - Morning Dew.mp3: BG=['bass guitar', 'drum machine'], MG=['electric piano', 'atmospheric pad'], FG=['electric guitar']
  - Black Pumas - Colors.mp3: BG=['bass guitar', 'kick drum', 'snare drum', 'hi-hats'], MG=['electric guitar chords', 'organ', 'acoustic guitar'], FG=['male lead vocal', 'electric guitar solo', 'backing vocals']
  - Chris Brown, Drake - No Guidance.mp3: BG=['sub bass', '808 kick', 'hi-hats'], MG=['electric piano chords', 'synth pad', 'synth stabs'], FG=['male lead vocal', 'vocal ad-libs', 'melodic vocal sample']
```

Performance on H100 (once model is loaded):
- vLLM throughput: ~5000 toks/s input, ~80 toks/s output

### Troubleshooting

If GPU memory is already in use from a previous run:
```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
```

TODO: Stem splitting based on instrument descriptions