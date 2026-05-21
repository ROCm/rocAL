#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$SCRIPT_DIR}"
LOG_DIR="${LOG_DIR:-/tmp/rocjpeg_decode_perf}"

GPU_COUNT="${1:-${GPU_COUNT:-1}}"
DATASET_LABEL="${DATASET_LABEL:-dataset}"

check_log() {
  if [ ! -f "$1" ]; then
    echo "ERROR: Missing log file: $1"
    exit 1
  fi
}

extract_count() {
  awk '/Total decoded images:/ {n=$4} END {if (n) printf "%d", n; else printf "0"}' "$1"
}

extract_decode_sec() {
  awk '
    /Total decoded images:/ {n=$4}
    /Average processing time per image/ {ms=$7}
    END {
      if (n && ms) printf "%.6f", (n * ms) / 1000.0;
      else printf "0.000000";
    }
  ' "$1"
}

echo "# Summary:"
echo ""
echo "### jpegdecodeperf sharded results with ${DATASET_LABEL}:"
echo ""
echo "GPU count: ${GPU_COUNT}"
echo ""

TOTAL_IMAGES=0
TOTAL_SEC=0
COUNT=0

for gpu in $(seq 0 $((GPU_COUNT - 1))); do
  LOG="$LOG_DIR/jpegdecodeperf_gpu${gpu}.log"
  check_log "$LOG"

  IMAGES=$(extract_count "$LOG")
  SEC=$(extract_decode_sec "$LOG")

  TOTAL_IMAGES=$(awk -v a="$TOTAL_IMAGES" -v b="$IMAGES" 'BEGIN {printf "%d", a + b}')
  TOTAL_SEC=$(awk -v a="$TOTAL_SEC" -v b="$SEC" 'BEGIN {printf "%.6f", a + b}')
  COUNT=$((COUNT + 1))

  printf "\tGPU/device %-3s images decoded: %10d decode time: %12.6f seconds\n" "$gpu" "$IMAGES" "$SEC"
done

AVG_SEC=$(awk -v total="$TOTAL_SEC" -v count="$COUNT" 'BEGIN {if (count > 0) printf "%.6f", total / count; else printf "0.000000"}')
MAX_SEC=$(awk '
  /Total decoded images:/ {n=$4}
  /Average processing time per image/ {
    ms=$7
    sec=(n * ms) / 1000.0
    if (sec > max) max=sec
  }
  END {
    printf "%.6f", max
  }
' "$LOG_DIR"/jpegdecodeperf_gpu*.log)

echo ""
echo "Decoded image count:"
echo ""
echo "        jpegdecodeperf total images decoded:             ${TOTAL_IMAGES}"

echo ""
echo "jpegdecodeperf sharded decode-time results:"
echo ""
printf "\tjpegdecodeperf average decode time:                 %12.6f seconds\n" "$AVG_SEC"
printf "\tjpegdecodeperf wall/max decode time:                %12.6f seconds\n" "$MAX_SEC"
echo ""
