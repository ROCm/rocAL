# rocAL validation report

**Mode:** PR run | baseline run
**Branch:** `<branch>` @ `<short sha>`
**Base:** `<base ref>` (merge-base `<short sha>`)
**Working tree:** clean | **DIRTY - N uncommitted file(s)**, listed below
**Date:** <YYYY-MM-DD>
**Machine:** backend `<HIP|CPU>`, deps: rocDecode `<y/n>`, rocJPEG `<y/n>`, FFmpeg `<y/n>`,
AUDIO_SUPPORT `<y/n>`, TurboJPEG `<y/n>`, LMDB `<y/n>`

> Delete whichever does not apply:
> **Baseline run** - no diff against the base branch. This validates the branch as it
> stands; it does not validate a change. Every coverage gap below applies.
> **PR run** - `<N>` file(s) changed vs `<base>`.

---

## ⚠ Coverage findings for this diff

*PR runs only. Omit this whole section for a baseline run. If Phase 0a flagged any touched
path as `ORPHAN`, `DISABLED`, `UNGATED` or `SMOKE`, it goes HERE, at the top - not in a
footer. A green run on an untested path is the most misleading result this skill can
produce.*

| Touched area | Nominal coverage | Status |
|---|---|---|
| `<area>` | `<test or suite>` | **ORPHAN - nothing runs it** |
| `<area>` | `<test or suite>` | SMOKE - proves it imports, not that it computes |

**What this means:** the gates below passed, but they did not exercise `<area>`. Verify it
by hand before merging, or say explicitly that it is unverified.

*If every touched area is `GATED`, say so in one line and drop the table.*

---

## Results

| # | Phase | Gate | Result | Numbers | vs baseline |
|---|---|---|---|---|---|
| 1 | Build | exit 0, deps recorded | PASS/FAIL | — | — |
| 2 | Install verified | `.text` match, 9 headers, data dirs | PASS/FAIL | — | — |
| 3 | ctest (cpp_api) | exit 0 **and** full expected set | PASS/FAIL | `<ran>/<expected>` | — |
| 4 | pybind | 6 tests discovered and pass | PASS/FAIL | `<n>/6` | — |
| 5a | python_api host | failing set == baseline | PASS/FAIL | `<p>/<f>/<t>` | `<delta>` |
| 5b | python_api hip | failing set == baseline | PASS/FAIL | `<p>/<f>/<t>` | `<delta>` |
| 5c | audio | `FAILED tests: 0` | PASS/FAIL/SKIP | `<n>` | — |

### Regressions

*New failures not in the baseline. Name each one. If none: "None."*

### Improvements

*Baseline failures that now pass. If any, note whether the baseline was re-pinned and in
which commit. If none: "None."*

---

## Not covered by this run

Summarised from `references/coverage-gaps.md` - this section is part of the result, not a
footnote.

- **7 of 15 `tests/cpp_api` directories have no working test:** `audio_tests`,
  `external_source`, `image_augmentation`, `multiple_dataloaders_test`,
  `voxel_augmentations_test`, plus `dataloader` and `dataloader_tf` whose `add_test` calls
  are commented out as non-functional.
- **pybind tests are import smoke tests** - 6/6 means the module loads, nothing more.
- **8 randomized augmentations are auto-passed** without comparison (`Snow`, `Rain`,
  `Jitter`, `SNPNoise`, `Fog`, `GaussianNoise`, `ShotNoise`, `Spatter`).
- **Not run by design:** `testAllScripts.sh` and `video_tests/testScript.sh` (both
  `sudo rm -rf`), `readers_test_file.sh` (no pass/fail of any kind), torch/JAX examples,
  `python_function_example.py` (needs `-DROCAL_PYTHON_FUNCTION=ON`).
- `<any phase skipped for a missing dependency - name the dependency and the lost tests>`

## Standing caveat

The `<19>` host and `<17>` hip python_api baseline failures are **known but not
root-caused**. `baselines.json` pins them so a different failure cannot hide inside the
same count; it does not bless them.

---

## Verdict

*One line, for pasting into the PR description.*

> Validated on `<branch>` @ `<sha>` against `<base>`: build + verified install, ctest
> `<n>/<n>`, pybind 6/6, python_api host `<f>`/hip `<f>` failures matching baseline. No
> regressions. Not covered: `<the short list>`.
