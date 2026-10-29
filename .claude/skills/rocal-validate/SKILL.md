---
name: rocal-validate
description: Run the full validation gate for a rocAL branch - build, verified install, ctest, pybind tests, and the python_api golden suite - then report results plus an explicit list of what was NOT covered. Works on a PR branch (scoping the report to the files that changed) or on develop as a plain regression run. Use before opening or updating a rocAL pull request, when asked to validate a rocAL branch or PR, or to check a rocAL change for regressions.
---

# rocAL validation gate

Turns "did I test this enough before opening the PR?" into one command with a fixed,
auditable set of gates.

Two supported modes, both first-class:

- **PR run** - validating a branch that has a diff against `develop`. The report names which
  test areas the diff actually lands on, including the ones with no working test.
- **Baseline run** - validating `develop` itself, or any branch with no diff. A plain
  regression check. The report says so, rather than implying it vetted a change.

## Why these gates and not `ctest && echo ok`

Four of rocAL's test entry points report success when they have tested nothing. The gates
below exist specifically to close those holes; do not simplify them away.

| Trap | What you see | What is true |
|---|---|---|
| Stale install | ctest: `100% tests passed` | Tests resolve rocAL via `find_library(... HINTS ${ROCM_PATH}/lib)`, so they re-validated the *previous* build |
| Missing dependency | ctest: `100% tests passed`, 10 tests | 13 tests were never registered; no GPU silently drops the run from 23 to 10 |
| pybind not installed | `No tests were found!!!`, exit **0** | Zero tests ran. `ctest` with no tests exits 0 |
| python_api failures | shell exit **0** | Nothing under `tests/python_api` sets an exit code on failure |

Run the phases in order. A failed gate stops the run - later phases are meaningless on a
bad install.

Throughout, `SK=.claude/skills/rocal-validate` and `<ts>` is one timestamp for the whole
run. Create `build/validate/<ts>/` first; `build*/` is already gitignored, so every
artifact this skill writes stays out of `git status`.

## Phase 0 - target, inputs, preflight

### 0a. What are we validating?

```bash
python3 -I $SK/scripts/resolve_target.py \
  [--target <pr-number|branch>] [--base develop] [--allow-checkout] \
  --json-out build/validate/<ts>/target.json
```

- **No `--target`** - validates whatever is checked out. This is how you run it on
  `develop`.
- **`--target 537`** - a bare number is a PR; the script runs `gh pr checkout 537`.
- **`--target some-branch`** - a branch name.

Switching branches is deliberately gated: the script refuses without `--allow-checkout`,
and refuses on a dirty tree regardless. Do not work around either. If the user is already
on the branch they want, run with no `--target` at all.

The script records branch, HEAD sha, merge-base against the base, the changed-file list,
and the dirty-tree state - so the report states exactly what was validated. It also maps
the diff onto test areas and **promotes any area the diff touches that has no working
test** (`ORPHAN`, `DISABLED`, `UNGATED`) or only an import smoke test (`SMOKE`).

The changed-file list includes **uncommitted** work (staged, unstaged and untracked). The
build and every suite validate the working tree, so scoping only committed changes would
drop the warnings for exactly the files under active edit.

Exit codes: `0` resolved, `1` hard error, **`2` the base could not be resolved**. On `2`
the committed diff is unknown - which is not the same as "no changes" - so the scoped
findings are incomplete. Fetch the base branch and re-run, or state plainly in the report
that diff scoping was unavailable. Do not report a `2` run as if it were a clean baseline.

That scoping is the point. rocAL's gaps are not uniform: a diff inside `color_augmentations`
is well covered, while an identically sized diff inside `audio_augmentations` is covered by
nothing in ctest. Carry that finding into the Phase 6 report prominently - a green run on an
untested path is the single most misleading result this skill can produce.

### 0b. Paths

Resolve two values, in this order: existing environment variable, then `defaults` in
`baselines.json` if present, then ask the user with a single `AskUserQuestion` carrying
both. Never hardcode machine-specific paths into the skill.

- `ROCM_PATH` - the ROCm install the tests link and load against
- `ROCAL_DATA_PATH` - the MIVisionX-data checkout holding the goldens

### 0c. Assert before doing any work

- both paths exist
- `$ROCAL_DATA_PATH/rocal_data/GoldenOutputsTensor/` exists - Phase 5 gates on it
- if the tree is dirty, **warn and continue**. Validating uncommitted work is legitimate;
  silently reporting it as though it were the committed branch is not. The report must say
  the tree was dirty and list the files.

## Phase 1 - build

```bash
set -o pipefail   # REQUIRED - without it the pipeline returns tee's status, so a
                  # failed compile exits 0 and carries stale artifacts into validation
cmake -S . -B build -DROCM_PATH="$ROCM_PATH" -DCMAKE_INSTALL_PREFIX="$ROCM_PATH" \
      2>&1 | tee build/validate/<ts>/configure.log \
  && cmake --build build -j"$(nproc)" 2>&1 | tee build/validate/<ts>/build.log
```

Configure and build are chained with `&&`: building on top of a failed configure just
reruns the previous configuration and produces a confusingly stale result.

Hard fail on non-zero. Check the status explicitly rather than trusting the absence of
error text on screen. From the configure output record which optional dependencies were
found - **rocDecode, rocJPEG, FFmpeg, libsndfile/AUDIO_SUPPORT, TurboJPEG, LMDB** - and
whether the backend is HIP or CPU. These pick the expected test counts in Phases 3 and 4,
and a dependency that disappeared since the last run is itself a finding worth reporting
even when every test passes.

## Phase 2 - install, then verify it landed

`sudo` is not available to a non-interactive tool call. Print exactly this and stop:

```
! cd <repo>/build && sudo make install
```

(The `!` prefix runs it in the user's own session.) When the user confirms, **verify rather
than trust**:

```bash
python3 -I $SK/scripts/verify_install.py \
  --build-dir build --rocm-path "$ROCM_PATH" --repo-root .
```

This compares **every ELF section** of the built and installed `librocal.so` and
`rocal_pybind*.so` - name, size and content - plus the dynamic entries, normalising away
only the RUNPATH carriers (`.dynstr`, `.dynamic`). It also diffs the 9 public headers, the
installed Python package, and the test-data directory.

Whole-file checksums cannot do this job: `make install` rewrites RUNPATH in place, so the
installed `.so` differs byte-for-byte at identical size and mtime even when the code is
identical. Hashing `.text` alone is also not enough - it covers host instructions only, so
a change confined to `.rodata` (lookup tables) or `.hip_fatbin` (embedded GPU device code)
would pass. Measured on a real `librocal.so.2.5.0`, a RUNPATH rewrite changes exactly one
section, `.dynstr`, leaving everything else comparable.

The installed **Python package** matters as much as the binaries: `rocAL_pybind/amd/`
installs to `$ROCM_PATH/lib/amd/`, and Phase 5 imports that installed copy. A change
confined to `fn.py` leaves every `.so` and header identical, so this is the only check that
catches it.

**Any mismatch aborts the run.** This is the gate that catches install skew.

## Phase 3 - ctest (tests/cpp_api)

```bash
python3 -I $SK/scripts/check_ctest.py \
  --build-dir build --baselines $SK/baselines.json \
  --backend <cpu|hip> [--rocdecode] [--rocjpeg] \
  --log build/validate/<ts>/ctest.log
```

Set the flags from what Phase 1 actually detected. They are **independent**, not a ladder:
`rocdecode_FOUND` and `rocjpeg_FOUND` are separate `if()` blocks nested inside the HIP
block, so 20, 21, 22 and 23 tests are all legitimate HIP configurations. Passing a
coarse "everything" tier on a box with only one of the two decoders would report the
other's tests as missing.

| backend | rocDecode | rocJPEG | expected |
|---|---|---|---|
| cpu | — | — | 10 |
| hip | no | no | 20 |
| hip | yes | no | 21 |
| hip | no | yes | 22 |
| hip | yes | yes | 23 |

Two gates, not one: the exit code must be clean **and** the discovered test set must match
the expected set. Fewer tests than expected is a FAIL naming the missing tests, not a pass.

Report plainly that ctest exercises **8 of the 15** `tests/cpp_api` directories and list
the 7 it does not touch (`references/coverage-gaps.md`), so the gap is visible rather than
implied.

## Phase 4 - pybind tests

`tests/pybind` is not in the top-level build - only `rocAL`, `rocAL_pybind` and
`tests/cpp_api` are `add_subdirectory`'d. Configure it as its own project:

```bash
cmake -S tests/pybind -B build/validate/<ts>/pybind -DROCM_PATH="$ROCM_PATH"
python3 -I $SK/scripts/check_ctest.py \
  --suite pybind --build-dir build/validate/<ts>/pybind \
  --baselines $SK/baselines.json --log build/validate/<ts>/pybind.log
```

Gate: exactly the 6 `rocal_pybind_test_*` tests are discovered and pass. **Zero discovered
is a FAIL** - it means pybind was not installed, and the plain reading of
`No tests were found!!!` plus exit 0 is the opposite of the truth.

These are import smoke tests; they assert nothing beyond the module loading. Say so in the
report so nobody reads 6/6 as functional coverage.

## Phase 5 - python_api golden suite

The suite must run from `tests/python_api` - its scripts resolve siblings relative to
their own cwd. Everything else in this skill is rooted at the repo, so pin absolute paths
up front and come back to the root before gating:

```bash
ROOT="$(git rev-parse --show-toplevel)"
OUT="$ROOT/build/validate/<ts>"

( cd "$ROOT/tests/python_api" \
  && export ROCAL_DATA_PATH PYTHONPATH="$ROCM_PATH/lib" \
            LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib" \
  && bash unit_tests.sh 0 2>&1 | tee "$OUT/python_api_host.log" )   # host
( cd "$ROOT/tests/python_api" && ... bash unit_tests.sh 1 2>&1 | tee "$OUT/python_api_hip.log" )  # hip
```

The subshells keep the `cd` from leaking. `set -o pipefail` is pointless here - the suite
exits 0 regardless - which is exactly why the gate below parses the log instead.

Then gate each backend, **from the repo root**, with absolute paths. The output directory
is created inside `tests/python_api`, so resolve it there, not from the cwd:

```bash
cd "$ROOT"
python3 -I "$ROOT/.claude/skills/rocal-validate/scripts/check_python_api.py" \
  --log "$OUT/python_api_host.log" --backend host \
  --baselines "$ROOT/.claude/skills/rocal-validate/baselines.json" \
  --output-dir "$ROOT/tests/python_api/<the rocal_python_unittest_output_folder_* this run produced>" \
  --golden-dir "$ROCAL_DATA_PATH/rocal_data/GoldenOutputsTensor"
```

The gate is on the failing **set**, not the count: a baseline of 19 host failures passing
with a *different* 19 is a regression, not a pass. It also checks the denominator, because
`image_comparison.py:104` iterates the output directory rather than the goldens - a case
that never produced a file vanishes from the total instead of failing.

If a baseline failure now passes, the script reports it as an improvement. Offer to pin the
new set with `--update`, but **never** run `--update` to make a red run go green.

If `AUDIO_SUPPORT` was built, also run `audio_unit_test.py` and gate on
`Number of FAILED tests: N`.

**Housekeeping.** The suite writes `rocal_python_unittest_output_folder_<ts>/` and
`rocal_unittest_log_file_<ts>_.log` into the cwd. Move both into `build/validate/<ts>/`
when the phase finishes. Both patterns are gitignored, but the repo already carries roughly
ten stale copies of each; leaving more behind compounds the mess.

## Phase 6 - report

Write `build/validate/<ts>/report.md` from `report_template.md` and summarise it in chat:

- mode (PR or baseline), branch, HEAD sha, base and merge-base, dirty-tree warning,
  detected dependency set
- a per-phase table: gate, result, numbers, delta vs baseline
- for a PR run, the **scoped coverage findings** from Phase 0a first - any touched path
  with no working test belongs at the top of the report, not in a footer
- the **"not covered by this run"** section from `references/coverage-gaps.md`; the gap is
  part of the result, not a footnote
- a one-line verdict suitable for pasting into the PR description

Always repeat the caveat that the baseline python_api failures are **known but not
root-caused**, so they do not quietly become permanent.

## Out of scope for v1

Not run, by design - each either needs `sudo rm -rf` or has nothing to gate on. Listed in
the report as a known gap rather than silently omitted; see `references/coverage-gaps.md`.
