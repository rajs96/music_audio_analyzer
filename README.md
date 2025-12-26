# Real-time and bulk instrument detection for downstream stem splitting and analysis

As of Christmas 2025:
This repo started out with ideas to help producers when they hear a song, and want to reproduce it or produce something similar.

I started off with the idea of advanced stem splitting to be able to feed as downstream data to to some sort of analysis. Currently this exists as a fast streaming pipeline to be able to take thousands of songs (potentially spanning many users). It does instrument detection as of now using Qwen Omni, but more to come! You can view the logic of the streaming pipeline here:


# Instrument Detection Pipeline Flow

This document walks through the complete flow of the instrument detector pipeline, step by step.

## Complete Pipeline Call Flow

### Step 1: Job Submission

```python
# In StreamingJobProducer._submit_job()
audio_file = Path("song.mp3")
audio_bytes = audio_file.read_bytes()
audio_ref = ray.put(audio_bytes)  # Store in Ray object store

job = InstrumentDetectJob(
    job_id="job_abc123",
    song_id="trk_xyz789",
    song_hash="sha256...",
    audio_ref=audio_ref,  # ObjectRef, not the bytes
    filename="song.mp3"
)
job_queue.put(job)
```

---

### Step 2: QueueStreamingDatasource reads job

```python
# In streaming_datasource.py - QueueStreamingDatasource.get_next_item()
item = self.queue.get(timeout=0.01)  # Returns InstrumentDetectJob

# Then item_to_row() is called (your job_to_row function)
row = {
    "job_id": "job_abc123",
    "song_id": "trk_xyz789",
    "song_hash": "sha256...",
    "filename": "song.mp3",
    "audio_ref": ObjectRef(...)  # Still an ObjectRef
}
```

---

### Step 3: Serialize for PyArrow

```python
# In streaming_datasource.py - make_block_generator()
serialized_row = _serialize_row_for_pyarrow(row)
# Result:
{
    "job_id": "job_abc123",
    "song_id": "trk_xyz789",
    "song_hash": "sha256...",
    "filename": "song.mp3",
    "audio_ref": b"__PICKLED_OBJREF__\x80\x04\x95..."  # Pickled ObjectRef
}

# Batch multiple rows, then:
table = pa.Table.from_pylist([serialized_row, ...])
yield table  # -> Ray Data Dataset
```

---

### Step 4: Ray Data applies first stage (AudioPreprocessorAgent)

```python
# Ray Data calls: preprocessor_callable(batch)
# batch = {"job_id": ["job_abc123"], "filename": ["song.mp3"], "audio_ref": [b"__PICKLED_..."]}

# In agent.py - AgentCallable._batch_to_items()
items = [_deserialize_row_from_pyarrow({
    "job_id": "job_abc123",
    "filename": "song.mp3",
    "audio_ref": b"__PICKLED_OBJREF__..."
})]
# Result after deserialization:
items = [{
    "job_id": "job_abc123",
    "filename": "song.mp3",
    "audio_ref": ObjectRef(...)  # Restored ObjectRef!
}]
```

---

### Step 5: AudioPreprocessorAgent.process_batch()

```python
# In audio_preprocessor.py
def process_batch(self, items):
    results = []
    for item in items:
        # Fetch audio bytes from Ray object store
        audio_bytes = ray.get(item["audio_ref"])  # Actual bytes now

        # Decode MP3 -> waveform
        waveform = self.decode_audio_bytes_to_waveform(audio_bytes, "mp3")
        # waveform.shape = (480000,) for 30 sec @ 16kHz

        # Store waveform in object store (memory efficient)
        waveform_ref = ray.put(waveform)

        results.append({
            "job_id": "job_abc123",
            "song_id": "trk_xyz789",
            "song_hash": "sha256...",
            "filename": "song.mp3",
            "waveform_ref": waveform_ref,  # ObjectRef to numpy array
            "error": None
        })
    return results
```

---

### Step 6: Serialize preprocessor output

```python
# In agent.py - AgentCallable._items_to_batch()
# Converts list of dicts -> dict of lists for Ray Data
output_batch = {
    "job_id": ["job_abc123"],
    "song_id": ["trk_xyz789"],
    "song_hash": ["sha256..."],
    "filename": ["song.mp3"],
    "waveform_ref": [ObjectRef(...)],  # Will be pickled again
    "error": [None]
}
```

---

### Step 7: Ray Data applies second stage (InstrumentDetectorAgent)

```python
# Ray Data calls: detector_callable(batch)
# Deserialize -> list of items with restored ObjectRefs

items = [{
    "job_id": "job_abc123",
    "filename": "song.mp3",
    "waveform_ref": ObjectRef(...),  # Restored
    "error": None
}]
```

---

### Step 8: InstrumentDetectorAgent.process_batch()

```python
# In instrument_detector.py
def process_batch(self, items):
    results = []
    valid_items = []
    valid_waveforms = []

    for item in items:
        # Check for upstream errors
        if item.get("error"):
            results.append({...error propagated...})
            continue

        # Fetch waveform from Ray object store
        waveform = ray.get(item["waveform_ref"])
        # waveform.shape = (480000,)

        valid_items.append(item)
        valid_waveforms.append(waveform)

    # Batch inference on GPU
    # waveforms = [np.array([...]), np.array([...]), ...]
    predictions = self.predict_batch_internal(valid_waveforms)
```

---

### Step 9: Model Inference (inside predict_batch_internal)

```python
def predict_batch_internal(self, waveforms):
    # Build conversations for each waveform
    conversations = []
    for waveform in waveforms:
        conversations.append([
            {"role": "system", "content": [{"type": "text", "text": "You are an expert..."}]},
            {"role": "user", "content": [{"type": "audio", "audio": waveform}]}
        ])

    # Tokenize
    inputs = self.processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move to GPU
    inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        output_ids = self.model.generate(**inputs, max_new_tokens=256)

    # Decode
    responses = self.processor.batch_decode(output_ids, skip_special_tokens=True)
    # responses = ["['drums', 'electric_guitar', 'piano']", ...]

    return responses
```

---

### Step 10: Parse and return results

```python
# Back in process_batch()
for item, prediction in zip(valid_items, predictions):
    # prediction = "['drums', 'electric_guitar', 'piano']"
    instruments = self._parse_instruments(prediction)
    # instruments = ['drums', 'electric_guitar', 'piano']

    results.append({
        "job_id": "job_abc123",
        "song_id": "trk_xyz789",
        "song_hash": "sha256...",
        "filename": "song.mp3",
        "instruments": ['drums', 'electric_guitar', 'piano'],
        "detected_at": 1735123456,
        "error": None
    })

return results
```

---

### Step 11: Results streamed to consumer

```python
# In test_streaming_pipeline.py
for batch in pipeline.stream(batch_size=1):
    # batch = {
    #     "job_id": ["job_abc123"],
    #     "filename": ["song.mp3"],
    #     "instruments": [['drums', 'electric_guitar', 'piano']],
    #     "detected_at": [1735123456],
    #     "error": [None]
    # }

    result = {k: batch[k][0] for k in batch.keys()}
    # result = {
    #     "job_id": "job_abc123",
    #     "filename": "song.mp3",
    #     "instruments": ['drums', 'electric_guitar', 'piano'],
    #     "error": None
    # }

    logger.info(f"Result: song.mp3 -> ['drums', 'electric_guitar', 'piano']")
```

---

## Visual Flow Diagram

```
+-------------------------------------------------------------------------+
|                           RAY OBJECT STORE                               |
|  +--------------+    +--------------+                                   |
|  | audio_bytes  |    |  waveform    |                                   |
|  | (MP3 data)   |    | (numpy arr)  |                                   |
|  +------+-------+    +------+-------+                                   |
|         | ObjectRef         | ObjectRef                                 |
+---------+-------------------+-------------------------------------------+
          |                   |
          v                   v
+-----------------+   +-----------------+   +-----------------+
|   Job Queue     |   |  Preprocessor   |   |    Detector     |
|                 |   |                 |   |                 |
| job.audio_ref --+-->| ray.get(ref)    |   | ray.get(ref)    |
|                 |   | decode MP3      |   | model.generate()|
|                 |   | ray.put(wav) ---+-->| parse output    |
|                 |   |                 |   |                 |
+-----------------+   +-----------------+   +-----------------+
                                                    |
                                                    v
                                            +-----------------+
                                            |    Results      |
                                            |                 |
                                            | instruments:    |
                                            | ['drums', ...]  |
                                            +-----------------+
```

---

## Data Flow Summary

| Stage | Data Location | Serialization |
|-------|---------------|---------------|
| Job Queue | ObjectRef in queue | Pickled for PyArrow |
| Preprocessor Input | Deserialized ObjectRef | ray.get() fetches bytes |
| Preprocessor Output | New ObjectRef for waveform | Pickled for PyArrow |
| Detector Input | Deserialized ObjectRef | ray.get() fetches waveform |
| Detector Output | Plain Python dict | Direct to consumer |

---

## Key Design Decisions

### 1. ObjectRef Serialization
- Ray `ObjectRef` cannot be directly stored in PyArrow tables
- We pickle ObjectRefs with a `__PICKLED_OBJREF__` prefix marker
- Automatically unpickled when reading batches in the agent callable

### 2. Waveform Storage
- Large numpy arrays (waveforms) are stored in Ray's object store
- Only the `ObjectRef` is passed through the pipeline
- This prevents memory bloat from duplicating arrays in PyArrow tables

### 3. Error Propagation
- Errors are propagated through the pipeline with an `error` field
- Failed items are not silently dropped
- Downstream stages skip errored items and propagate the error

### 4. Thread Safety
- The `_items_read` counter uses a threading lock for parallelism > 1
- Stop events use `threading.Event()` for safe cross-thread signaling

