# What a green `rocal-validate` run does NOT tell you

Verified against `develop` at `85088402`. Every claim below cites the file that makes it
true, so it can be re-checked when the tree moves.

This file exists because the gaps are not obvious from the outside: `ctest` prints
`100% tests passed` whether it ran 23 tests or 10, and the suites that cover the remaining
directories either cannot run unattended or report no pass/fail at all.

## 1. ctest covers 8 of the 15 `tests/cpp_api` directories

23 tests are registered (`tests/cpp_api/CMakeLists.txt`), drawn from:

`basic_test`, `checkpoint_test`, `dataloader_multithread`, `performance_tests`,
`performance_tests_with_depth`, `serialization_test`, `unit_tests`, `video_tests`

### The 7 with no working test

| Directory | Why nothing runs it |
|---|---|
| `audio_tests` | no `add_test` at all |
| `external_source` | no `add_test` at all |
| `image_augmentation` | no `add_test` at all |
| `multiple_dataloaders_test` | no `add_test` at all |
| `voxel_augmentations_test` | no `add_test` at all |
| `dataloader` | `add_test` commented out - "non-functional with database" (`CMakeLists.txt:118`) |
| `dataloader_tf` | `add_test` commented out - "non-functional with tf dataset" (`CMakeLists.txt:158`) |

Each would need its own build-and-run recipe from its README to be wired up. That is the
obvious follow-up work once the core flow is trusted.

## 2. A missing dependency shrinks the run instead of failing it

Tests are registered inside `if()` blocks keyed on `BACKEND STREQUAL "HIP"`,
`rocdecode_FOUND` and `rocjpeg_FOUND`:

| Condition | Tests registered | Cumulative |
|---|---|---|
| always | 10 | 10 |
| `+ BACKEND == HIP` | 10 | 20 |
| `+ rocdecode_FOUND` | 1 | 21 |
| `+ rocjpeg_FOUND` | 2 | 23 |

On a machine with no GPU the suite runs 10 tests and prints `100% tests passed`. This is
why Phase 3 gates on the discovered test *set* and not the exit code.

## 3. `ctest` exits 0 when it finds no tests

Verified directly: an empty CMake project with `enable_testing()` prints
`No tests were found!!!` and `echo $?` gives `0`.

`tests/pybind/CMakeLists.txt` wraps all 6 `add_test` calls in
`if(Python3_FOUND AND ROCAL_PYBIND_SCRIPTS)`, where `ROCAL_PYBIND_SCRIPTS` is set only when
`${ROCM_PATH}/lib/amd/rocal` exists. Without an installed pybind the whole suite silently
evaporates into a passing run.

## 4. Six of the eight pybind tests are import smoke tests

`rocal_pybind_test_{decoders,functions,pipelines,randoms,readers,types}` execute their
scripts and check the exit code. Green means the module loads and the scripts run - not
that any computation is correct.

The two added since are real tests: `rocal_pybind_test_pipeline_scope` (#580) and
`rocal_pybind_test_rejected_operator` (#582) assert actual behaviour. So "8/8 passed" is
six smoke tests plus two assertions, not eight assertions.

## 5. An exit code alone still cannot gate `tests/python_api`

**#616 fixed the worst of this.** Both image comparators now end in
`exit(1 if failed_case_count else 0)`, and `unit_tests.sh` ends with that call, so the
suite's status propagates. The old "nothing sets an exit code" claim is obsolete - do not
repeat it.

What remains:

- `unit_tests.sh:4-8` still guards a missing `ROCAL_DATA_PATH` with a bare `exit`, which
  exits **0**. The suite prints `Need to export ROCAL_DATA_PATH` and reports success
  having run nothing.
- The status is 1 on any run with a known failure - every run today - so it cannot
  distinguish the same 9 known failures from a different 9 of the same size.

## 6. The python_api denominator can shrink silently

`image_comparison.py:104` iterates `rocal_output_dir_list` - the **outputs produced**, not
the goldens. A case that never produced an output file is not in that listing, so it is
never counted at all: it disappears from the denominator rather than failing.

Two further ways a case leaves the buckets without failing:

- `:202` `"Skipping the testcase as file not found"` - counted in the total, in neither
  passed nor failed, so `passed + failed < total`.
- `:203` `"File not found in ref_output_folder"` - no matching golden; not counted anywhere.

Phase 5 therefore gates on the total, on `passed + failed == total`, and on the failing
set - not on the failure count.

## 7. Randomized augmentations are auto-passed

`image_comparison.py:101` lists `Snow, Rain, Jitter, SNPNoise, Fog, GaussianNoise,
ShotNoise, Spatter`. For these the script increments `passed_case_count` without comparing
anything (`:114-119`). They cannot fail. Treat them as "ran without crashing".

## 8. Suites this skill deliberately does not run

| Entry point | Why not |
|---|---|
| `tests/cpp_api/unit_tests/testAllScripts.sh` | self-builds every case from scratch - a much longer job than these gates. The largest suite in the repo and the best candidate for an opt-in `--full` flag. **No longer dangerous:** #615 anchored it to its own location and it now does a plain `rm -rf "$build_dir"`, not `sudo rm -rf ./build/*`. |
| `tests/cpp_api/video_tests/testScript.sh` | same - also anchored by #615, plain `rm -rf "$BUILD_DIR"` |
| `tests/python_api/readers_test_file.sh` | no goldens, no comparison, no exit code; which pipelines run is edited in-file |
| torch / JAX example scripts | examples, not gated tests |
| `python_function_example.py` | needs `-DROCAL_PYTHON_FUNCTION=ON`, off by default |

### 8a. Most of `tests/python_api` is not run at all

Phase 5 runs `unit_tests.sh`, which invokes exactly **two** scripts - `unit_test.py` and
`image_comparison.py` (plus `parse_config.py`, imported by the former) - and Phase 5 runs
`audio_unit_test.py` separately when `AUDIO_SUPPORT` was built.

Every other file in that directory is a standalone example that nothing in this skill
executes:

```
caffe2_reader.py  caffe_reader.py  cifar10_reader.py  coco_reader.py  decoder.py
external_source_reader.py  jax_classification_reader.py  numpy_reader.py  pipeline.py
pipeline_checkpoint_example.py  pipeline_serialize_test.py  prefetch_queue_depth/
pytorch_classification_reader.py  pytorch_diffusion_loader.py
pytorch_multi_channel_loader.py  tf_classification_reader.py  tf_detection_reader.py
video_pipeline.py  web_dataset_reader.py
```

Living under `tests/` is not coverage. `resolve_target.py` keeps an **allow-list** of the
entry points a phase actually invokes, so anything added to this directory later defaults
to `UNGATED` rather than inheriting a blanket "test code, therefore gated" claim.

## 9. A green ctest says almost nothing about augmentations or most readers

Two gates look far broader than they are. Both were flagged by @kirancv22 on PR #583 and
are now reported `PARTIAL`, never `GATED`:

| Gate | Looks like | Actually runs |
|---|---|---|
| `unit_tests_{cpu,gpu,gray}` | all image augmentations | `test_case=1` only - `rocalCropResize`, 1 of ~103 cases in the switch. The argv is hardcoded in `tests/cpp_api/CMakeLists.txt`. |
| `basic_test_*` | the readers | the plain file path only. `basic_test.cpp` calls `rocalCreateLabelReader` and `rocalCreateTextFileBasedLabelReader` and nothing else. |

The golden suite does cover coco, tf, caffe, caffe2, mxnet and web_dataset via
`--reader-type`. **Covered by neither**, and reported `ORPHAN`: the CIFAR-10 reader, the
NumPy reader, the external-source reader and the video *sequence* reader.

## 10. The python_api baseline failures are known, not accepted

Host fails 9/151 and hip 10/152, measured 2026-11-20 on gfx1100. **None are root-caused.** `baselines.json` pins the
names so a *different* failure cannot hide inside the same count. It does not bless them.
Every report repeats this, so they do not quietly become permanent.

**Several are missing goldens, not mismatches.** `image_comparison.py` counts a case with
no reference image as a failure, and MIVisionX-data has none for some recently added
augmentations - 5 of the 9 on host (`Dilate_*`, `Erode_*`, `Posterize_gray`) and 2 of the
10 on hip. Goldens are per-backend, so a name can lack one on host and have one on hip.
`no_golden` in `baselines.json` lists them. Mint the goldens and these stop being noise.

**Re-measure, never re-derive.** When #587 disabled ten cases it was tempting to patch
`139 - 10 = 129`. The measured total is **151**: develop had also *added* cases
(`Posterize`, `Solarize`, `LUT`, `JpegCompressionDistortion`, ...). Arithmetic on a
baseline yields a number that looks authoritative and is wrong.
