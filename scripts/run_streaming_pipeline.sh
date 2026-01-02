#!/bin/bash
# Run the streaming instrument detection pipeline with configurable options

set -e

# =============================================================================
# Job Producer Config
# =============================================================================
DATA_DIR="audio_files"
TOTAL_JOBS=50
MIN_DELAY=100
MAX_DELAY=2000
BURST_PROB=0.2
BURST_MIN=3
BURST_MAX=10

# =============================================================================
# Datasource Config
# =============================================================================
DS_BATCH=8
DS_PARALLELISM=1

# =============================================================================
# Preprocessor Config
# =============================================================================
NUM_PREPROCESSORS=4
PREP_BATCH=8
PREP_CPUS=1.0
PREP_CONCURRENCY=1

# =============================================================================
# Detector Config
# =============================================================================
NUM_DETECTORS=1
DETECTOR_BATCH=4
DETECTOR_GPUS=1.0
DETECTOR_CONCURRENCY=1

# =============================================================================
# Model Config
# =============================================================================
MODEL="Qwen/Qwen3-Omni-30B-A3B-Thinking"
CACHE_DIR="/app/cache"
MODELS_DIR="/app/models"
SKIP_CACHE=false
DTYPE="bfloat16"

# =============================================================================
# vLLM Config
# =============================================================================
USE_VLLM=false
USE_COT=false
TENSOR_PARALLEL_SIZE=""
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=8

# =============================================================================
# Build the command
# =============================================================================

CMD="python test/test_streaming_pipeline.py"

# Job producer options
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --total-jobs $TOTAL_JOBS"
CMD="$CMD --min-delay $MIN_DELAY"
CMD="$CMD --max-delay $MAX_DELAY"
CMD="$CMD --burst-prob $BURST_PROB"
CMD="$CMD --burst-min $BURST_MIN"
CMD="$CMD --burst-max $BURST_MAX"

# Datasource options
CMD="$CMD --ds-batch $DS_BATCH"
CMD="$CMD --ds-parallelism $DS_PARALLELISM"

# Preprocessor options
CMD="$CMD --num-preprocessors $NUM_PREPROCESSORS"
CMD="$CMD --prep-batch $PREP_BATCH"
CMD="$CMD --prep-cpus $PREP_CPUS"
CMD="$CMD --prep-concurrency $PREP_CONCURRENCY"

# Detector options
CMD="$CMD --num-detectors $NUM_DETECTORS"
CMD="$CMD --detector-batch $DETECTOR_BATCH"
CMD="$CMD --detector-gpus $DETECTOR_GPUS"
CMD="$CMD --detector-concurrency $DETECTOR_CONCURRENCY"

# Model options
CMD="$CMD --model $MODEL"
CMD="$CMD --cache-dir $CACHE_DIR"
CMD="$CMD --models-dir $MODELS_DIR"
CMD="$CMD --dtype $DTYPE"

if [ "$SKIP_CACHE" = true ]; then
    CMD="$CMD --skip-cache"
fi

# vLLM options
if [ "$USE_VLLM" = true ]; then
    CMD="$CMD --use-vllm"
fi

if [ "$USE_COT" = true ]; then
    CMD="$CMD --use-cot"
fi

if [ -n "$TENSOR_PARALLEL_SIZE" ]; then
    CMD="$CMD --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
fi

CMD="$CMD --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
CMD="$CMD --max-model-len $MAX_MODEL_LEN"
CMD="$CMD --max-num-seqs $MAX_NUM_SEQS"

# =============================================================================
# Run
# =============================================================================

echo "Running: $CMD"
exec $CMD
