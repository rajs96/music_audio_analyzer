import ray
from ray import serve
import requests
from pathlib import Path
from src.instrument_detect.producer import InstrumentDetectJobProducer
from src.instrument_detect.job_queue import InstrumentDetectJobQueue
from loguru import logger

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    serve.start(detached=False)
    queue = InstrumentDetectJobQueue.options(name="instrument_detect_queue").remote(
        max_size=1000
    )
    producer = InstrumentDetectJobProducer.bind(queue)
    serve.run(producer)

    logger.info("Service is running at http://127.0.0.1:8000")
    logger.info("Press Ctrl+C to stop...")

    # Try uploading some files
    BASE_URL = "http://127.0.0.1:8000"
    AUDIO_FILES_DIR = "../audio_files"

    # Get all files in the audio_files directory
    audio_dir = Path(AUDIO_FILES_DIR)
    audio_files = list(audio_dir.glob("*"))

    # Filter out directories if any
    audio_files = [f for f in audio_files if f.is_file()]

    logger.info(f"Found {len(audio_files)} files to upload:")
    for f in audio_files:
        logger.info(f"  - {f.name}")

    files_list = []
    for audio_file in audio_files:
        files_list.append(
            ("files", (audio_file.name, open(audio_file, "rb"), "audio/mpeg"))
        )

    # Upload
    response = requests.post(f"{BASE_URL}/v1/upload", files=files_list)

    # Clean up
    for _, (_, file_handle, _) in files_list:
        file_handle.close()

    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {response.text}")

    # --- Measure queue size after ---
    after_size = ray.get(queue.size.remote())
    logger.info(f"Queue size AFTER upload: {after_size}")

    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        serve.shutdown()
