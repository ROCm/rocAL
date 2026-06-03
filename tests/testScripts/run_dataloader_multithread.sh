#!/bin/bash

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$SCRIPT_DIR}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
LOG_DIR="${LOG_DIR:-/tmp/rocjpeg_decode_perf}"

GPU_COUNT="${1:-${GPU_COUNT:-1}}"
SHARD_COUNT="$GPU_COUNT"

if [ -z "${DATASET:-}" ]; then
  echo "ERROR: DATASET is not set."
  echo "Example: export DATASET=/path/to/image_dataset"
  exit 1
fi

export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROCM_PATH/lib${PYTHONPATH:+:$PYTHONPATH}"
export ROCJPEG_DECODER_CREATE_LOG="${ROCJPEG_DECODER_CREATE_LOG:-1}"

if [ -z "${ROCAL_CPP_BIN:-}" ]; then
  echo "ERROR: ROCAL_CPP_BIN is not set."
  echo "Example: export ROCAL_CPP_BIN=/path/to/dataloader_multithread"
  exit 1
fi

ROCAL_PY_BENCH="${ROCAL_PY_BENCH:-$WORKSPACE/rocal_decode_call_bench.py}"

mkdir -p "$LOG_DIR"

CPP_ROCJPEG_LOG="${CPP_ROCJPEG_LOG:-$LOG_DIR/rocjpeg_${GPU_COUNT}gpu.log}"
CPP_TURBOJPEG_LOG="${CPP_TURBOJPEG_LOG:-$LOG_DIR/turbojpeg_${GPU_COUNT}gpu.log}"

PY_ROCJPEG_LOG="${PY_ROCJPEG_LOG:-$LOG_DIR/py_rocjpeg_${GPU_COUNT}gpu.log}"
PY_TURBOJPEG_LOG="${PY_TURBOJPEG_LOG:-$LOG_DIR/py_turbojpeg_${GPU_COUNT}gpu.log}"

echo "WORKSPACE: $WORKSPACE"
echo "DATASET: $DATASET"
echo "GPU_COUNT: $GPU_COUNT"
echo "SHARD_COUNT: $SHARD_COUNT"
echo "ROCM_PATH: $ROCM_PATH"
echo "LOG_DIR: $LOG_DIR"
echo "ROCJPEG_DECODER_CREATE_LOG: $ROCJPEG_DECODER_CREATE_LOG"
echo "ROCAL_CPP_BIN: $ROCAL_CPP_BIN"
echo "ROCAL_PY_BENCH: $ROCAL_PY_BENCH"
echo ""

echo "============================================================"
echo "C++ rocAL + rocJPEG"
echo "============================================================"
"$ROCAL_CPP_BIN" "$DATASET" "$GPU_COUNT" "$SHARD_COUNT" 1024 1024 32 0 0 4 4 2>&1 | tee "$CPP_ROCJPEG_LOG"

echo ""
echo "============================================================"
echo "C++ rocAL + TurboJPEG, one run only"
echo "============================================================"
"$ROCAL_CPP_BIN" "$DATASET" "$GPU_COUNT" "$SHARD_COUNT" 1024 1024 32 0 0 0 4 2>&1 | tee "$CPP_TURBOJPEG_LOG"

echo ""
echo "============================================================"
echo "Python rocAL + rocJPEG"
echo "============================================================"
python3 "$ROCAL_PY_BENCH" --path "$DATASET" --device gpu --batch-size 32 --num-threads 4 --device-id 0 --num-gpus "$GPU_COUNT" --num-shards "$SHARD_COUNT" 2>&1 | tee "$PY_ROCJPEG_LOG"

echo ""
echo "============================================================"
echo "Python rocAL + TurboJPEG, one run only"
echo "============================================================"
python3 "$ROCAL_PY_BENCH" --path "$DATASET" --device cpu --batch-size 32 --num-threads 4 --device-id 0 --num-gpus "$GPU_COUNT" --num-shards "$SHARD_COUNT" 2>&1 | tee "$PY_TURBOJPEG_LOG"

echo ""
echo "Done. Logs written to:"
echo "  $CPP_ROCJPEG_LOG"
echo "  $CPP_TURBOJPEG_LOG"
echo "  $PY_ROCJPEG_LOG"
echo "  $PY_TURBOJPEG_LOG"
