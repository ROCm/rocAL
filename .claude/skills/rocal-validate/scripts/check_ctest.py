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

Usage:
    check_ctest.py --build-dir build --baselines baselines.json \\
                   [--tier hip_rocdecode_rocjpeg | --expect-names a,b,c] [--log ctest.log]
    check_ctest.py --suite pybind --build-dir build/validate/pybind --baselines baselines.json

Exit 0 = exit code clean AND the expected set ran. Exit 1 otherwise.
"""

import argparse
import json
import os
import re
import subprocess
import sys


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
    ap.add_argument("--tier", help="cpp_api only: cpu_only | hip | hip_rocdecode_rocjpeg")
    ap.add_argument("--expect-names", help="comma-separated override of the expected set")
    ap.add_argument("--log", help="path to a ctest run log; if omitted, this script runs ctest")
    args = ap.parse_args()

    with open(args.baselines) as fh:
        base = json.load(fh)

    # ---- expected set -------------------------------------------------------
    if args.expect_names:
        expected = sorted({n.strip() for n in args.expect_names.split(",") if n.strip()})
    elif args.suite == "pybind":
        expected = sorted(base["pybind"]["expected"])
    else:
        tiers = base["ctest"]["tiers"]
        if not args.tier:
            ap.error("cpp_api needs --tier (or --expect-names)")
        wanted = {
            "cpu_only": ["always"],
            "hip": ["always", "hip"],
            "hip_rocdecode_rocjpeg": ["always", "hip", "rocdecode", "rocjpeg"],
        }.get(args.tier)
        if wanted is None:
            ap.error(f"unknown tier {args.tier!r}")
        expected = sorted(n for key in wanted for n in tiers[key])

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
