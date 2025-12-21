from ray import serve
import hashlib
import ray
import time
import uuid
from loguru import logger
from fastapi import FastAPI, UploadFile, File

from src.instrument_detect.data_classes import InstrumentDetectJob
from src.instrument_detect.job_queue import InstrumentDetectJobQueue

app = FastAPI()


@serve.deployment(max_ongoing_requests=100)
@serve.ingress(app)
class InstrumentDetectJobProducer:
    """Can upload files and create jobs to split files"""

    def __init__(self, queue: InstrumentDetectJobQueue):
        self.queue = queue

    @app.get("/v1/status")
    async def status(self):
        return {"status": "ok"}

    @app.post("/v1/upload")
    async def upload(self, files: list[UploadFile] = File(...)):
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        created_at = int(time.time())

        songs = []
        for f in files:
            audio_bytes = await f.read()
            song_id = f"trk_{uuid.uuid4().hex[:12]}"
            song_hash = hashlib.sha256(audio_bytes).hexdigest()

            audio_ref = ray.put(
                audio_bytes
            )  # store in ray object store of deployment, later s3

            detect_job = InstrumentDetectJob(
                job_id=job_id,
                created_at=created_at,
                song_id=song_id,
                song_hash=song_hash,
                audio_ref=audio_ref,
                filename=f.filename,
            )
            logger.info(f"Enqueuing job {detect_job.job_id} for filename {f.filename}")
            self.queue.enqueue.remote(detect_job)

        return {"ok": True, "job_id": job_id}


if __name__ == "__main__":
    # Start / connect Ray locally
    ray.init(ignore_reinit_error=True)

    # Start Serve (HTTP on :8000 by default)
    serve.start(detached=False)

    # Create the queue actor
    queue = InstrumentDetectJobQueue.options(name="instrument_detect_queue").remote(
        max_size=1000
    )

    # Deploy the producer
    serve.run(InstrumentDetectJobProducer.bind(queue))
    # Keep the script alive
    print("Service is running at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop...")
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        serve.shutdown()
