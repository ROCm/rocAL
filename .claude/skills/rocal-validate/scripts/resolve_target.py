#!/usr/bin/env python3
"""Resolve what this validation run is actually validating, and scope the coverage gaps.

Two modes, both legitimate:

  * PR branch  - `--target 537` or `--target my-branch`. There is a diff against the base,
                 so the run can say which test areas the change actually touches.
  * base branch - no target, or the checkout already is `develop`. There is no diff; this
                 is a baseline/regression run. Say so rather than implying PR coverage.

The scoping matters because rocAL's coverage gaps are not uniform. Seven `tests/cpp_api`
directories have no `add_test` at all. A diff that stays inside `color_augmentations` is
well covered; the identical-sized diff inside `audio_augmentations` is covered by nothing
in ctest. A generic "here are the known gaps" footer buries that difference - this script
promotes the gaps the diff actually lands on.

Usage:
    resolve_target.py [--target <pr-number|branch>] [--base develop] [--json-out FILE]

Switching branches is gated: it refuses on a dirty tree and never switches without
--allow-checkout, because clobbering the user's working tree is not this skill's business.
"""

import argparse
import json
import os
import re
import subprocess
import sys

# path prefix -> (area description, gating status)
# Status vocabulary:
#   GATED      - a phase in this skill will fail if it breaks
#   SMOKE      - a test exists but only proves the module imports
#   ORPHAN     - tests/cpp_api directory with no add_test; nothing runs it
#   DISABLED   - add_test exists but is commented out in CMakeLists.txt
#   UNGATED    - a script exists but reports no pass/fail of any kind
COVERAGE_MAP = [
    ("rocAL/source/augmentations/audio_augmentations", "audio augmentations",
     [("audio_unit_test.py (only when AUDIO_SUPPORT was built)", "GATED"),
      ("tests/cpp_api/audio_tests", "ORPHAN")]),
    ("rocAL/source/augmentations", "image augmentations",
     [("tests/python_api golden suite", "GATED"),
      ("ctest unit_tests_{cpu,gpu,gray}", "GATED"),
      ("tests/cpp_api/image_augmentation", "ORPHAN")]),
    ("rocAL/source/decoders/video", "video decoding",
     [("ctest video_tests_ffmpeg / video_tests_rocdecode", "GATED")]),
    ("rocAL/source/decoders/audio", "audio decoding",
     [("audio_unit_test.py (only when AUDIO_SUPPORT was built)", "GATED"),
      ("tests/cpp_api/audio_tests", "ORPHAN")]),
    ("rocAL/source/decoders", "image decoding",
     [("ctest basic_test_* incl. rocjpeg tier", "GATED")]),
    ("rocAL/source/loaders/video", "video loading",
     [("ctest video_tests_*", "GATED")]),
    ("rocAL/source/loaders/audio", "audio loading",
     [("tests/cpp_api/audio_tests", "ORPHAN")]),
    ("rocAL/source/loaders", "image loading",
     [("ctest dataloader_multithread_{cpu,gpu}", "GATED"),
      ("tests/cpp_api/dataloader", "DISABLED"),
      ("tests/cpp_api/multiple_dataloaders_test", "ORPHAN")]),
    ("rocAL/source/readers", "readers",
     [("tests/python_api/readers_test_file.sh", "UNGATED"),
      ("tests/cpp_api/dataloader_tf", "DISABLED"),
      ("ctest basic_test_* (file reader path only)", "GATED")]),
    ("rocAL/source/pipeline", "core pipeline / graph",
     [("every phase", "GATED"),
      ("ctest checkpoint_save/restore, serialization_test_*", "GATED")]),
    ("rocAL/source/meta_data", "metadata readers",
     [("ctest unit_tests_one_hot", "GATED"),
      ("tests/python_api coco/caffe/tf readers", "GATED")]),
    ("rocAL/source/parameters", "random parameter generators",
     [("rocal_pybind_test_randoms", "SMOKE"),
      ("tests/python_api randomized augmentations are auto-passed", "UNGATED")]),
    ("rocAL/source/device", "HIP device layer",
     [("every GPU-tier ctest case", "GATED")]),
    ("rocAL/source/api", "C API surface",
     [("all cpp_api tests", "GATED")]),
    ("rocAL/include/api", "public headers",
     [("install header diff (phase 2)", "GATED"), ("all cpp_api tests", "GATED")]),
    ("rocAL_pybind", "Python bindings",
     [("tests/pybind import tests", "SMOKE"),
      ("tests/python_api golden suite", "GATED")]),
    ("tests/", "the test suites themselves", [("n/a - this is test code", "GATED")]),
]

# Paths that genuinely need no test coverage. Anything else that fails to map is
# reported, because a silently unmapped source file gets no scoping at all - which is
# exactly the blind spot this script exists to remove.
IGNORABLE = re.compile(
    r"(^|/)(docs?|\.github|cmake|rocAL-setup\.py|CHANGELOG\.md|README\.md|LICENSE)"
    r"|\.(md|rst|txt|ya?ml|json|toml|cfg|in|png|jpg|svg)$"
)

ORPHAN_DIRS = [
    "audio_tests", "dataloader", "dataloader_tf", "external_source",
    "image_augmentation", "multiple_dataloaders_test", "voxel_augmentations_test",
]


def scope_paths(paths):
    """Map changed paths onto test areas.

    Returns (implicated, unmapped). `implicated` is de-duplicated and carries the gating
    status of each area; `unmapped` is every non-ignorable path no rule matched.
    """
    implicated, seen, unmapped = [], set(), []
    for path in paths:
        # rocAL/include/ mirrors rocAL/source/ one-for-one, so a header change implicates
        # the same tests as the matching .cpp. include/api is excluded: it has its own
        # entry because the phase 2 install header diff gates it specifically.
        probe = path
        if path.startswith("rocAL/include/") and not path.startswith("rocAL/include/api"):
            probe = "rocAL/source/" + path[len("rocAL/include/"):]

        for prefix, area, areas in COVERAGE_MAP:
            if probe.startswith(prefix):
                for name, status in areas:
                    key = (area, name, status)
                    if key not in seen:
                        seen.add(key)
                        implicated.append(
                            {"area": area, "covered_by": name, "status": status})
                break
        else:
            if not IGNORABLE.search(path):
                unmapped.append(path)
    return implicated, unmapped


def git(*args, cwd=None, check=True):
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{res.stderr.strip()}")
    return res.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="PR number or branch name; omit to use the current checkout")
    ap.add_argument("--base", default="develop", help="base branch to diff against")
    ap.add_argument("--allow-checkout", action="store_true",
                    help="permit switching branches (refused on a dirty tree regardless)")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    root = git("rev-parse", "--show-toplevel")
    dirty_files = [l for l in git("status", "--porcelain", cwd=root).splitlines() if l.strip()]

    # ---- optional checkout --------------------------------------------------
    switched = None
    if args.target:
        current = git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
        is_pr = bool(re.fullmatch(r"\d+", args.target))
        want = f"PR #{args.target}" if is_pr else f"branch {args.target}"
        already = (not is_pr) and current == args.target
        if already:
            print(f"  already on {want}")
        elif not args.allow_checkout:
            raise SystemExit(
                f"Refusing to switch to {want} without --allow-checkout.\n"
                f"Currently on '{current}'. Either check it out yourself and re-run without\n"
                f"--target, or pass --allow-checkout to let this script do it."
            )
        elif dirty_files:
            raise SystemExit(
                f"Refusing to switch to {want}: the working tree has "
                f"{len(dirty_files)} uncommitted change(s).\n"
                "Commit or set them aside first - switching would clobber them."
            )
        else:
            if is_pr:
                subprocess.run(["gh", "pr", "checkout", args.target], cwd=root, check=True)
            else:
                git("checkout", args.target, cwd=root)
            switched = f"{current} -> {args.target}"
            print(f"  checked out {want} (was {current})")

    branch = git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    sha = git("rev-parse", "HEAD", cwd=root)
    dirty_files = [l for l in git("status", "--porcelain", cwd=root).splitlines() if l.strip()]

    # ---- base resolution ----------------------------------------------------
    # Prefer origin/<base>. A local 'develop' is routinely stale, and diffing against a
    # stale base invents a diff out of commits that are already upstream - which would
    # scope the coverage findings to files the branch never touched.
    base_ref = None
    for cand in (f"origin/{args.base}", args.base):
        if subprocess.run(["git", "rev-parse", "--verify", "--quiet", cand],
                          cwd=root, capture_output=True).returncode == 0:
            base_ref = cand
            break

    merge_base = changed = None
    if base_ref:
        merge_base = git("merge-base", "HEAD", base_ref, cwd=root, check=False) or None
        if merge_base:
            out = git("diff", "--name-only", f"{merge_base}..HEAD", cwd=root, check=False)
            changed = [l for l in out.splitlines() if l.strip()]

    is_baseline_run = not changed

    # ---- scope the gaps -----------------------------------------------------
    implicated, unmapped = scope_paths(changed or [])

    # ---- report -------------------------------------------------------------
    print(f"  branch     : {branch}")
    print(f"  HEAD       : {sha[:12]}")
    print(f"  base       : {base_ref or '<not found - cannot diff>'}")
    if dirty_files:
        print(f"  tree       : DIRTY ({len(dirty_files)} file(s)) - the report must say so")
        for l in dirty_files[:20]:
            print(f"               {l}")
    else:
        print("  tree       : clean")

    if is_baseline_run:
        print("\n  Mode: BASELINE RUN - no diff against the base branch.")
        print("  This validates the branch as it stands; it does not validate a change.")
        print("  Every coverage gap applies; none can be ruled out as untouched.")
    else:
        print(f"\n  Mode: PR RUN - {len(changed)} file(s) changed vs {base_ref}")
        blind = [i for i in implicated if i["status"] in ("ORPHAN", "DISABLED", "UNGATED")]
        smoke = [i for i in implicated if i["status"] == "SMOKE"]
        if blind:
            print("\n  *** THIS DIFF TOUCHES CODE WITH NO WORKING TEST ***")
            for i in blind:
                print(f"      [{i['status']:8}] {i['area']}: {i['covered_by']}")
            print("      A green run does NOT mean these paths were exercised.")
        if smoke:
            print("\n  Smoke-only coverage (proves it imports, not that it computes):")
            for i in smoke:
                print(f"      [{i['status']:8}] {i['area']}: {i['covered_by']}")
        gated = [i for i in implicated if i["status"] == "GATED"]
        if gated:
            print("\n  Properly gated by this run:")
            for i in gated:
                print(f"      {i['area']}: {i['covered_by']}")
        if unmapped:
            print("\n  UNMAPPED source files - no coverage rule matched these:")
            for p in unmapped:
                print(f"      {p}")
            print("      Work out by hand which suite exercises them, and either extend")
            print("      COVERAGE_MAP in this script or say in the report that they are")
            print("      unverified. Do not let them pass silently.")
        if not implicated and not unmapped:
            print("  No mapped test area - the diff is outside rocAL/ and rocAL_pybind/")
            print("  (docs, CI, packaging). The suites still run as a regression check.")

    payload = {
        "branch": branch, "head": sha, "base": base_ref, "merge_base": merge_base,
        "switched": switched, "dirty": dirty_files,
        "mode": "baseline" if is_baseline_run else "pr",
        "changed_files": changed or [], "implicated": implicated,
        "unmapped": unmapped, "orphan_dirs": ORPHAN_DIRS,
    }
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
