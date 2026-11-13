#!/usr/bin/env python3
"""Gate a ctest run on the *set* of tests discovered, not just the exit code.

Why this exists
---------------
tests/cpp_api/CMakeLists.txt registers tests inside if() blocks keyed on BACKEND==HIP,
rocdecode_FOUND and rocjpeg_FOUND. A missing dependency therefore SHRINKS the run instead
of failing it: no GPU drops 23 tests to 10 and still prints "100% tests passed". And a
project with no tests at all exits 0 ("No tests were found!!!"). Exit code alone cannot
distinguish "everything passed" from "almost nothing ran".

This script compares discovered test names against the tiered inventory in baselines.json
and names the specific tests that went missing.

The expected set is built from independent dependency flags rather than a fixed ladder,
because rocdecode_FOUND and rocjpeg_FOUND are separate if() blocks
(tests/cpp_api/CMakeLists.txt:251-270). A HIP box with rocDecode but no rocJPEG registers
21 tests and a box with rocJPEG but no rocDecode registers 22 - both are valid, and a
three-rung ladder of cpu/hip/everything would wrongly flag one as missing tests and the
other as having unexpected ones.

It also gates the nested build dirs before anything compiles. Each cpp_api case is built
by `ctest --build-and-test` into a dir that persists under build/tests/cpp_api/ and is
reused on every later run, and those sub-projects cache ROCM_PATH without FORCE - so the
first value a dir ever saw stays pinned no matter what the environment says afterwards.
Reuse build/ across two ROCm installs and the suite links the old one while phase 2
verified the new one.

Usage:
    check_ctest.py --build-dir build --baselines baselines.json \\
                   --rocm-path "$ROCM_PATH" \\
                   --backend hip [--rocdecode] [--rocjpeg] [--log ctest.log]
    check_ctest.py --build-dir build --baselines baselines.json --rocm-path "$ROCM_PATH" \\
                   --backend cpu [--clean-stale-cache]
    check_ctest.py --suite pybind --build-dir build/validate/pybind --baselines baselines.json

Pass --expect-names to override the computed set entirely.

Exit 0 = exit code clean AND the expected set ran. Exit 1 otherwise.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from glob import glob


def cache_value(cache_path, key):
    """Read one entry out of a CMakeCache.txt, or None if absent/unreadable."""
    try:
        with open(cache_path, errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(("//", "#")) or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.split(":", 1)[0] == key:
                    return value
    except OSError:
        return None
    return None


def stale_nested_caches(build_dir, rocm_path):
    """Nested cpp_api build dirs whose cached ROCM_PATH no longer matches.

    Each cpp_api case is built by `ctest --build-and-test <src> <bin>` into a dir under
    build/tests/cpp_api/<case>/ that PERSISTS and is reused by every later run. Those
    sub-projects take ROCm from

        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "...")

    with no FORCE (tests/cpp_api/basic_test/CMakeLists.txt:28-33), and `set(... CACHE ...)`
    without FORCE is a no-op once the entry exists. The FIRST ROCM_PATH such a dir ever saw
    is therefore pinned for its lifetime: exporting a different one changes nothing, and
    `-DROCM_PATH` on the top-level configure never reaches these sub-projects at all.

    That makes phase 0b's export necessary but not sufficient. Switch ROCm installs while
    reusing build/ and phase 2 verifies install A while the suite silently links and loads
    install B - the exact stale-install failure this skill exists to catch, arriving by a
    route that looks clean from the outside.

    Detected by reading the value back out of each nested cache rather than by stamping a
    file, so it reports what the sub-projects will actually use.
    """
    want = os.path.realpath(rocm_path)
    stale = []
    pattern = os.path.join(build_dir, "tests", "cpp_api", "*", "CMakeCache.txt")
    for cache in sorted(glob(pattern)):
        got = cache_value(cache, "ROCM_PATH")
        if got and os.path.realpath(got) != want:
            stale.append((os.path.dirname(cache), got))
    return stale


def discover(build_dir):
    """ctest -N lists registered tests without running them."""
    res = subprocess.run(
        ["ctest", "-N"], cwd=build_dir, capture_output=True, text=True,
    )
    names = re.findall(r"^\s*Test\s+#\d+:\s+(\S+)\s*$", res.stdout, re.M)
    return names, res.stdout


def parse_results(log_text):
    """Pull pass/fail counts and failing names out of a ctest run log."""
    passed = failed = total = None
    m = re.search(r"(\d+)%\s+tests passed,\s+(\d+)\s+tests failed out of\s+(\d+)", log_text)
    if m:
        failed, total = int(m.group(2)), int(m.group(3))
        passed = total - failed
    failing = re.findall(r"^\s*\d+\s+-\s+(\S+)\s+\((?:Failed|Timeout|SEGFAULT|Subprocess aborted|Exception)[^)]*\)",
                         log_text, re.M)
    return passed, failed, total, failing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--suite", choices=["cpp_api", "pybind"], default="cpp_api")
    ap.add_argument("--backend", choices=["cpu", "hip"],
                    help="cpp_api only: the BACKEND the tests were configured with")
    ap.add_argument("--rocdecode", action="store_true",
                    help="rocdecode_FOUND was true at configure time")
    ap.add_argument("--rocjpeg", action="store_true",
                    help="rocjpeg_FOUND was true at configure time")
    ap.add_argument("--expect-names", help="comma-separated override of the expected set")
    ap.add_argument("--log", help="path to a ctest run log; if omitted, this script runs ctest")
    ap.add_argument("--rocm-path",
                    help="the ROCM_PATH this run is validating; required for cpp_api, which "
                         "reuses nested build dirs that pin the first value they ever saw")
    ap.add_argument("--clean-stale-cache", action="store_true",
                    help="delete nested cpp_api build dirs pinned to a different ROCM_PATH "
                         "instead of failing")
    args = ap.parse_args()

    with open(args.baselines) as fh:
        base = json.load(fh)

    # ---- nested-cache staleness (before anything builds) --------------------
    # Must come first: ctest -N only lists, but the run below BUILDS each case in those
    # nested dirs, so a pinned ROCM_PATH has to be caught before a single object is
    # compiled against the wrong install.
    if args.suite == "cpp_api":
        if not args.rocm_path:
            ap.error(
                "cpp_api needs --rocm-path. The nested per-case build dirs cache ROCM_PATH "
                "without FORCE, so a reused build/ keeps linking whichever ROCm install it "
                "saw first - and this gate cannot be skipped silently.")
        stale = stale_nested_caches(args.build_dir, args.rocm_path)
        if stale and args.clean_stale_cache:
            for d, _ in stale:
                shutil.rmtree(d, ignore_errors=True)
            print(f"  cleaned    : {len(stale)} nested build dir(s) pinned to a stale "
                  f"ROCM_PATH; they will reconfigure against {args.rocm_path}")
        elif stale:
            print("CTEST GATE FAILED: stale nested build dirs.\n")
            print("  These were configured against a different ROCM_PATH than this run is")
            print(f"  validating ({args.rocm_path}), and `set(ROCM_PATH ... CACHE PATH)`")
            print("  without FORCE will NOT pick up the new value - the cached one wins:\n")
            for d, got in stale:
                print(f"    {d}\n        pinned to {got}")
            print("\n  Running now would build and link every one of those cases against the")
            print("  OLD install while phase 2 verified the new one. Re-run with")
            print("  --clean-stale-cache, or delete the directories listed above.")
            return 1

    # ---- expected set -------------------------------------------------------
    if args.expect_names:
        expected = sorted({n.strip() for n in args.expect_names.split(",") if n.strip()})
    elif args.suite == "pybind":
        expected = sorted(base["pybind"]["expected"])
    else:
        tiers = base["ctest"]["tiers"]
        if not args.backend:
            ap.error("cpp_api needs --backend cpu|hip (or --expect-names)")
        # rocDecode and rocJPEG only ever register tests inside the HIP block, so their
        # flags are ignored on a CPU backend rather than silently inflating the set.
        wanted = ["always"]
        if args.backend == "hip":
            wanted.append("hip")
            if args.rocdecode:
                wanted.append("rocdecode")
            if args.rocjpeg:
                wanted.append("rocjpeg")
        elif args.rocdecode or args.rocjpeg:
            print("  note: --rocdecode/--rocjpeg ignored on a CPU backend - those tests "
                  "are registered only inside the HIP block")
        expected = sorted(n for key in wanted for n in tiers[key])
        print(f"  config     : backend={args.backend}"
              f" rocdecode={args.rocdecode} rocjpeg={args.rocjpeg}")

    # ---- discovered set -----------------------------------------------------
    discovered, _ = discover(args.build_dir)
    discovered_set = sorted(set(discovered))

    print(f"  expected   : {len(expected)} test(s)")
    print(f"  discovered : {len(discovered_set)} test(s)")

    problems = []

    if not discovered_set:
        problems.append(
            "ZERO tests discovered. ctest reports 'No tests were found!!!' and still exits 0, "
            "so this would otherwise read as a pass. For tests/pybind this means rocAL pybind "
            "is not installed at $ROCM_PATH/lib/amd/rocal."
        )

    missing = sorted(set(expected) - set(discovered_set))
    extra = sorted(set(discovered_set) - set(expected))
    if missing:
        problems.append("MISSING (silently not registered): " + ", ".join(missing))
    if extra:
        problems.append(
            "UNEXPECTED (new tests added upstream? update baselines.json): " + ", ".join(extra)
        )

    # ---- run results --------------------------------------------------------
    if args.log and os.path.isfile(args.log):
        log_text = open(args.log, errors="replace").read()
    else:
        res = subprocess.run(
            ["ctest", "--output-on-failure"], cwd=args.build_dir,
            capture_output=True, text=True,
        )
        log_text = res.stdout + res.stderr
        if args.log:
            with open(args.log, "w") as fh:
                fh.write(log_text)

    passed, failed, total, failing = parse_results(log_text)
    if total is None:
        problems.append("could not parse a ctest summary line from the run output")
    else:
        print(f"  ran        : {total}, passed {passed}, failed {failed}")
        if failed:
            problems.append(f"{failed} test(s) FAILED: " + ", ".join(failing or ["<unparsed>"]))
        if total != len(discovered_set):
            problems.append(
                f"ran {total} but {len(discovered_set)} were registered - a filter or a crash "
                "cut the run short"
            )

    if problems:
        print("\nCTEST GATE FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print("\nctest gate passed: the full expected set ran and every test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
