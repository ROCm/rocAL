# rocJPEG Decode Performance Harness

This folder contains a small manual performance and validation harness for
testing rocAL image decode behavior with rocJPEG and TurboJPEG. It is intended
for developer and PR-reviewer runs that exercise the rocJPEG split-decoder path
through both the C++ and Python rocAL entry points.

Location in rocAL:

```text
tests/testScripts/
```

## Files

```text
rocal_decode_call_bench.py
run_dataloader_multithread.sh
```

### `rocal_decode_call_bench.py`

Python rocAL benchmark for exercising `fn.readers.file` and
`fn.decoders.image`. It can run one shard in-process or launch one worker
process per shard for multi-GPU tests.

The script prints:

- total decoded image count derived from filtered `.jpg` / `.jpeg` files
- total processing time
- average per-image time
- public rocAL internal timing fields when available

The public rocAL `TimingInfo` API is used for timing values. Image-count
reporting comes from the dataset file count, following symlinks.

Expected rocJPEG usage:

```bash
python3 rocal_decode_call_bench.py \
  --path <dataset_dir> \
  --device gpu \
  --batch-size 32 \
  --num-threads 4 \
  --device-id 0 \
  --num-gpus <gpu_count> \
  --num-shards <shard_count>
```

For a TurboJPEG run through the CPU decode path:

```bash
python3 rocal_decode_call_bench.py \
  --path <dataset_dir> \
  --device cpu \
  --batch-size 32 \
  --num-threads 4 \
  --device-id 0 \
  --num-gpus <gpu_count> \
  --num-shards <shard_count>
```

### `run_dataloader_multithread.sh`

Main rocAL benchmark driver. It runs four cases and stores logs in `LOG_DIR`:

```text
C++ rocAL + rocJPEG
C++ rocAL + TurboJPEG
Python rocAL + rocJPEG
Python rocAL + TurboJPEG
```

The rocJPEG split-decoder path is the rocAL default behavior for rocJPEG decode.
The benchmark driver does not set a split toggle environment variable.

Expected usage:

```bash
export DATASET=/path/to/image_dataset
export ROCAL_CPP_BIN=/path/to/dataloader_multithread
export ROCM_PATH=/opt/rocm
export LOG_DIR=/tmp/rocjpeg_decode_perf

./run_dataloader_multithread.sh <gpu_count>
```

## Environment Variables

### Required

```bash
export DATASET=/path/to/image_dataset
export ROCAL_CPP_BIN=/path/to/dataloader_multithread
```

`DATASET` points to the input image directory. `ROCAL_CPP_BIN` points to the
compiled rocAL `dataloader_multithread` binary used by the C++ benchmark runs.

### Common Optional Variables

```bash
export DATASET_LABEL=dataset
export GPU_COUNT=1
export LOG_DIR=/tmp/rocjpeg_decode_perf
export WORKSPACE=/path/to/rocAL/tests/testScripts
export ROCAL_PY_BENCH=$WORKSPACE/rocal_decode_call_bench.py
```

`WORKSPACE` defaults to the directory containing the shell script, so it usually
does not need to be exported. `LOG_DIR` is shared by all generated logs.

### rocAL/ROCm Runtime Variables

```bash
export ROCM_PATH=/opt/rocm
export ROCJPEG_DECODER_CREATE_LOG=1
```

The scripts use `ROCM_PATH` to set `LD_LIBRARY_PATH` and `PYTHONPATH`. If
`ROCM_PATH` is not set, it defaults to `/opt/rocm`.

## Build Notes

This harness is manual/performance-oriented and is not wired into the regular
CTest flow. It is intended for explicit developer or PR-reviewer runs on systems
with the needed dataset, rocAL build, rocJPEG support, and GPU configuration.

If installing this with rocAL test assets, include the shell and Python files as
test support files rather than compiling them.

This rocAL PR includes the rocAL-side changes only. Any rocJPEG decoder creation
logging patch that targets rocJPEG belongs in the rocJPEG repo and is not
included here.

## Typical Workflow

```bash
cd /path/to/rocAL/tests/testScripts

export DATASET=/path/to/image_dataset
export DATASET_LABEL=my_dataset
export ROCAL_CPP_BIN=/path/to/dataloader_multithread
export ROCM_PATH=/opt/rocm
export LOG_DIR=/tmp/rocjpeg_decode_perf

./run_dataloader_multithread.sh 1
```

## Output Logs

The scripts write logs to `LOG_DIR`. If `LOG_DIR` is not set, it defaults to:

```text
/tmp/rocjpeg_decode_perf
```

rocAL benchmark logs:

```text
$LOG_DIR/rocjpeg_<N>gpu.log
$LOG_DIR/turbojpeg_<N>gpu.log
$LOG_DIR/py_rocjpeg_<N>gpu.log
$LOG_DIR/py_turbojpeg_<N>gpu.log
```

## Script Group

These scripts are placed under:

```text
tests/testScripts
```

This keeps the harness with other test support scripts while the files remain
focused on rocJPEG-backed image decode performance in rocAL, including sharded
multi-GPU decode runs through rocAL and TurboJPEG benchmark runs.

## Example to Set Env Vars Before Running Any Script

For a local workspace where this folder is under `/workspace/rocAL/tests/testScripts`
and the test dataset is `/workspace/test_1300_files/train`, use:

```bash
export WORKSPACE=/workspace/rocAL/tests/testScripts
cd "$WORKSPACE"

export DATASET=/workspace/test_1300_files/train
export DATASET_LABEL=test_1300_files
export ROCM_PATH=/opt/rocm
export LOG_DIR=/tmp/rocjpeg_decode_perf
export ROCAL_CPP_BIN=/workspace/rocAL/build/tests/cpp_api/dataloader_multithread_manual/dataloader_multithread
```

Then run the main rocAL benchmark:

```bash
./run_dataloader_multithread.sh 1
```
