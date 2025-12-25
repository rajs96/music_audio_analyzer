# Deprecated - use src.instrument_detect.pipeline instead
#
# The streaming pipeline is now implemented as:
#   - PreprocessorActor (preprocessor.py) - CPU workers that decode audio
#   - DetectorActor (detector.py) - GPU worker that runs Qwen inference
#   - InstrumentDetectPipeline (pipeline.py) - Orchestrates the actors
#
# Example usage:
#
#   from src.instrument_detect.pipeline import create_pipeline
#   from src.instrument_detect.job_queue import create_job_queue
#
#   job_queue = create_job_queue()
#   pipeline = create_pipeline(job_queue, num_preprocessors=4)
#
#   for result in pipeline.iter_results():
#       print(f"{result.filename}: {result.instruments}")
