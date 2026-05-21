#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$SCRIPT_DIR}"
LOG_DIR="${LOG_DIR:-/tmp/rocjpeg_decode_perf}"

GPU_COUNT="${1:-${GPU_COUNT:-1}}"

CPP_ROCJPEG_OFF_LOG="${CPP_ROCJPEG_OFF_LOG:-$LOG_DIR/rocjpeg_split_off_${GPU_COUNT}gpu.log}"
CPP_ROCJPEG_ON_LOG="${CPP_ROCJPEG_ON_LOG:-$LOG_DIR/rocjpeg_split_on_${GPU_COUNT}gpu.log}"
CPP_TURBOJPEG_LOG="${CPP_TURBOJPEG_LOG:-$LOG_DIR/turbojpeg_${GPU_COUNT}gpu.log}"

PY_ROCJPEG_OFF_LOG="${PY_ROCJPEG_OFF_LOG:-$LOG_DIR/py_rocjpeg_split_off_${GPU_COUNT}gpu.log}"
PY_ROCJPEG_ON_LOG="${PY_ROCJPEG_ON_LOG:-$LOG_DIR/py_rocjpeg_split_on_${GPU_COUNT}gpu.log}"
PY_TURBOJPEG_LOG="${PY_TURBOJPEG_LOG:-$LOG_DIR/py_turbojpeg_${GPU_COUNT}gpu.log}"

DATASET_LABEL="${DATASET_LABEL:-dataset}"

check_log() {
  if [ ! -f "$1" ]; then
    echo "ERROR: Missing log file: $1"
    exit 1
  fi
}

print_line() {
  label="$1"
  sec="$2"
  printf "\t%-52s %12.6f seconds\n" "$label" "$sec"
}

extract_cpp_total_count() {
  awk '/Remaining images/ {total += $3; count += 1} END {if (count > 0) printf "%d", total; else printf "0"}' "$1"
}

extract_py_total_count() {
  awk '/Total decoded images:/ {total += $4; count += 1} END {if (count > 0) printf "%d", total; else printf "0"}' "$1"
}

extract_cpp_avg_decode_sec() {
  awk '
    /Decode   time:/ {
      total += $3 / 1000000.0;
      count += 1;
    }
    END {
      if (count > 0) printf "%.6f", total / count;
      else printf "0.000000";
    }
  ' "$1"
}

print_cpp_per_gpu_decode() {
  awk '
    /For shard_id:/ {
      shard=$3;
      gsub(/[^0-9]/, "", shard);
    }
    /Decode   time:/ {
      sec=$3 / 1000000.0;
      if (shard == "") shard=count;
      printf "\tGPU/shard %-3s decode time: %12.6f seconds\n", shard, sec;
      count += 1;
      shard="";
    }
  ' "$1"
}

extract_py_avg_decode_sec() {
  awk '
    /Average rocAL internal decode time across/ {val=$NF}
    /^rocAL internal decode time/ {
      total += $NF;
      count += 1;
      single=$NF;
    }
    END {
      if (val != "") printf "%.6f", val;
      else if (count > 1) printf "%.6f", total / count;
      else if (single != "") printf "%.6f", single;
      else printf "0.000000";
    }
  ' "$1"
}

print_py_per_gpu_decode() {
  awk '
    /GPU\/device id:/ {
      gpu=$NF;
    }
    /Shard id:/ {
      shard=$NF;
    }
    /^rocAL internal decode time/ {
      sec=$NF;
      if (gpu != "" && shard != "") {
        printf "\tGPU/device %-3s shard %-3s decode time: %12.6f seconds\n", gpu, shard, sec;
      } else {
        printf "\tdecode time: %12.6f seconds\n", sec;
      }
      gpu="";
      shard="";
    }
  ' "$1"
}

calc_improvement() {
  awk -v off="$1" -v on="$2" 'BEGIN {
    if (off > 0) printf "%.2f", ((off - on) / off) * 100.0;
    else printf "0.00";
  }'
}

calc_speedup() {
  awk -v off="$1" -v on="$2" 'BEGIN {
    if (on > 0) printf "%.2f", off / on;
    else printf "0.00";
  }'
}

check_log "$CPP_ROCJPEG_OFF_LOG"
check_log "$CPP_ROCJPEG_ON_LOG"
check_log "$CPP_TURBOJPEG_LOG"
check_log "$PY_ROCJPEG_OFF_LOG"
check_log "$PY_ROCJPEG_ON_LOG"
check_log "$PY_TURBOJPEG_LOG"

CPP_ROCJPEG_OFF_COUNT=$(extract_cpp_total_count "$CPP_ROCJPEG_OFF_LOG")
CPP_ROCJPEG_ON_COUNT=$(extract_cpp_total_count "$CPP_ROCJPEG_ON_LOG")
CPP_TURBOJPEG_COUNT=$(extract_cpp_total_count "$CPP_TURBOJPEG_LOG")

PY_ROCJPEG_OFF_COUNT=$(extract_py_total_count "$PY_ROCJPEG_OFF_LOG")
PY_ROCJPEG_ON_COUNT=$(extract_py_total_count "$PY_ROCJPEG_ON_LOG")
PY_TURBOJPEG_COUNT=$(extract_py_total_count "$PY_TURBOJPEG_LOG")

CPP_ROCJPEG_OFF_AVG=$(extract_cpp_avg_decode_sec "$CPP_ROCJPEG_OFF_LOG")
CPP_ROCJPEG_ON_AVG=$(extract_cpp_avg_decode_sec "$CPP_ROCJPEG_ON_LOG")
CPP_TURBOJPEG_AVG=$(extract_cpp_avg_decode_sec "$CPP_TURBOJPEG_LOG")

PY_ROCJPEG_OFF_AVG=$(extract_py_avg_decode_sec "$PY_ROCJPEG_OFF_LOG")
PY_ROCJPEG_ON_AVG=$(extract_py_avg_decode_sec "$PY_ROCJPEG_ON_LOG")
PY_TURBOJPEG_AVG=$(extract_py_avg_decode_sec "$PY_TURBOJPEG_LOG")

CPP_IMPROVEMENT=$(calc_improvement "$CPP_ROCJPEG_OFF_AVG" "$CPP_ROCJPEG_ON_AVG")
CPP_SPEEDUP=$(calc_speedup "$CPP_ROCJPEG_OFF_AVG" "$CPP_ROCJPEG_ON_AVG")

PY_IMPROVEMENT=$(calc_improvement "$PY_ROCJPEG_OFF_AVG" "$PY_ROCJPEG_ON_AVG")
PY_SPEEDUP=$(calc_speedup "$PY_ROCJPEG_OFF_AVG" "$PY_ROCJPEG_ON_AVG")

echo "# Summary:"
echo ""
echo "### With ${DATASET_LABEL}:"
echo ""
echo "GPU count: ${GPU_COUNT}"
echo ""

echo "Decoded image count:"
echo ""
echo "        C++ rocAL+rocJPEG OFF images decoded:           ${CPP_ROCJPEG_OFF_COUNT}"
echo "        C++ rocAL+rocJPEG ON images decoded:            ${CPP_ROCJPEG_ON_COUNT}"
echo "        C++ rocAL+TurboJPEG images decoded:             ${CPP_TURBOJPEG_COUNT}"
echo "        PY  rocAL+rocJPEG OFF images decoded:           ${PY_ROCJPEG_OFF_COUNT}"
echo "        PY  rocAL+rocJPEG ON images decoded:            ${PY_ROCJPEG_ON_COUNT}"
echo "        PY  rocAL+TurboJPEG images decoded:             ${PY_TURBOJPEG_COUNT}"

echo ""
echo "### C++ rocAL sample decode-time results:"
echo ""
echo "Without rocAL patch solution"
print_cpp_per_gpu_decode "$CPP_ROCJPEG_OFF_LOG"
print_line "rocAL+rocJPEG C++ average decode time:" "$CPP_ROCJPEG_OFF_AVG"

echo ""
echo "With rocAL patch solution"
print_cpp_per_gpu_decode "$CPP_ROCJPEG_ON_LOG"
print_line "rocAL+rocJPEG C++ average decode time:" "$CPP_ROCJPEG_ON_AVG"

echo ""
echo "TurboJPEG one run"
print_cpp_per_gpu_decode "$CPP_TURBOJPEG_LOG"
print_line "rocAL+TurboJPEG C++ average decode time:" "$CPP_TURBOJPEG_AVG"

echo ""
echo "### Python rocAL benchmark decode-time results:"
echo ""
echo "Without rocAL patch solution"
print_py_per_gpu_decode "$PY_ROCJPEG_OFF_LOG"
print_line "rocAL+rocJPEG PY average decode time:" "$PY_ROCJPEG_OFF_AVG"

echo ""
echo "With rocAL patch solution"
print_py_per_gpu_decode "$PY_ROCJPEG_ON_LOG"
print_line "rocAL+rocJPEG PY average decode time:" "$PY_ROCJPEG_ON_AVG"

echo ""
echo "TurboJPEG one run"
print_py_per_gpu_decode "$PY_TURBOJPEG_LOG"
print_line "rocAL+TurboJPEG PY average decode time:" "$PY_TURBOJPEG_AVG"

echo ""
echo "The rocAL patch solution decode-time enhancements when used:"
echo ""
printf "\tC++ rocAL+rocJPEG enhancement: %s%% decode-time reduction, speedup around %sx\n" "$CPP_IMPROVEMENT" "$CPP_SPEEDUP"
printf "\tPY  rocAL+rocJPEG enhancement: %s%% decode-time reduction, speedup around %sx\n" "$PY_IMPROVEMENT" "$PY_SPEEDUP"
echo ""
