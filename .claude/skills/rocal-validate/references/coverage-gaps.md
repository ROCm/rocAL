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

## 4. The pybind tests are import smoke tests

`rocal_pybind_test_{decoders,functions,pipelines,randoms,readers,types}` execute their
scripts and check the exit code. 6/6 green means the module loads and the scripts run - not
that any computation is correct.

## 5. Nothing under `tests/python_api` sets an exit code

- `unit_tests.sh:4-8` guards a missing `ROCAL_DATA_PATH` with a bare `exit`, which exits
  **0**. The suite prints `Need to export ROCAL_DATA_PATH` and reports success.
- `image_comparison.py` never calls `sys.exit()` on failure; it logs and returns.

So `bash unit_tests.sh 0 && echo ok` prints `ok` whether 0 or 139 cases failed.

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
| `tests/cpp_api/unit_tests/testAllScripts.sh` | runs `sudo rm -rf ./build/*` and self-builds; cannot run unattended. The largest suite in the repo - the best candidate for an opt-in `--full` flag once the core flow is trusted. |
| `tests/cpp_api/video_tests/testScript.sh` | also `sudo rm -rvf build*` |
| `tests/python_api/readers_test_file.sh` | no goldens, no comparison, no exit code; which pipelines run is edited in-file |
| torch / JAX example scripts | examples, not gated tests |
| `python_function_example.py` | needs `-DROCAL_PYTHON_FUNCTION=ON`, off by default |

## 9. The python_api baseline failures are known, not accepted

As of 2026-10-28, host fails 19/139 and hip fails 17/139. **None are root-caused.**
`baselines.json` pins them so a *different* failure cannot hide inside the same count. It
does not bless them. Every report repeats this, so they do not quietly become permanent.
