#!/usr/bin/env python3
"""Gate a ctest run on the *set* of tests that should have run, not just the exit code.

Why this exists
---------------
tests/cpp_api/CMakeLists.txt registers tests inside if() blocks. A missing dependency
therefore SHRINKS the run instead of failing it: no GPU drops 23 tests to 10 and still
prints "100% tests passed". A project with no tests at all exits 0 ("No tests were
found!!!"). Exit code alone cannot tell "everything passed" from "almost nothing ran".

The expected set is DERIVED from CMakeLists.txt, never pinned
-------------------------------------------------------------
An earlier version of this script kept the inventory in baselines.json. It went stale
twice in three weeks - #580/#582 added two pybind tests, and the cpp_api video tests were
re-gated from `rocdecode_FOUND` onto `ROCAL_VIDEO_TESTS_ENABLED` /
`ROCAL_VIDEO_ROCDECODE_AVAILABLE` - and a stale inventory fails in the worst direction:
it reports real tests as UNEXPECTED and mis-tiers the rest.

So the inventory is parsed out of the CMakeLists that defines it, together with the if()
conditions each add_test sits under. The only hand-maintained part is the mapping from a
CMake condition to a command-line flag, and an unrecognised condition is a HARD ERROR
rather than a guess - so re-gating a test on something new cannot silently mis-tier it.

Usage:
    check_ctest.py --build-dir build --source-dir tests/cpp_api \\
                   --rocm-path "$ROCM_PATH" --backend hip \\
                   [--rocjpeg] [--video-tests] [--video-rocdecode] \\
                   [--clean-stale-cache] [--log ctest.log]
    check_ctest.py --suite pybind --build-dir build/validate/pybind \\
                   --source-dir tests/pybind

Exit 0 = exit code clean AND the expected set ran. Exit 1 otherwise.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from glob import glob

# CMake condition -> the argparse dest that decides whether it holds.
#
# Hand-maintained on purpose, and deliberately small. Anything guarding an add_test that
# is not in here aborts the run: that is the signal that upstream re-gated a test and this
# skill has not caught up. Guessing instead would reintroduce exactly the silent drift
# that retiring the pinned inventory was meant to remove.
KNOWN_CONDITIONS = {
    '"${BACKEND}" STREQUAL "HIP"': "hip",
    "rocjpeg_FOUND": "rocjpeg",
    "ROCAL_VIDEO_TESTS_ENABLED": "video_tests",
    "ROCAL_VIDEO_ROCDECODE_AVAILABLE": "video_rocdecode",
    "Python3_FOUND AND ROCAL_PYBIND_SCRIPTS": "pybind_scripts",
}

IF_RE = re.compile(r"^if\s*\((.*)\)\s*$")
ELSEIF_RE = re.compile(r"^elseif\s*\((.*)\)\s*$")
ELSE_RE = re.compile(r"^else\s*\(")
ENDIF_RE = re.compile(r"^endif\s*\(")
ADD_TEST_RE = re.compile(r"^add_test\s*\(")
NAME_RE = re.compile(r"\bNAME\s+([A-Za-z0-9_]+)")


def registered_tests(cmakelists):
    """-> [(test_name, [conditions])] for every live add_test in the file.

    Commented-out add_test calls are skipped, which is the point: tests/cpp_api has
    several that are commented out as non-functional, and they must not be expected.
    """
    if not os.path.isfile(cmakelists):
        raise SystemExit(f"cannot read {cmakelists} - is --source-dir correct?")

    stack, found = [], []
    buf, buf_conds, depth = None, None, 0
    for raw in open(cmakelists, errors="replace"):
        line = raw.strip()
        if line.startswith("#"):
            continue

        # Inside an add_test(...) call: buffer until the parens balance, then read NAME
        # off the whole call. tests/cpp_api writes most of them as
        #     add_test(
        #       NAME
        #         basic_test_cpu
        # so NAME and its value are on different lines and a per-line regex sees neither.
        if buf is not None:
            buf.append(line)
            depth += line.count("(") - line.count(")")
            if depth <= 0:
                m = NAME_RE.search(" ".join(buf))
                if m:
                    found.append((m.group(1), buf_conds))
                buf, buf_conds, depth = None, None, 0
            continue

        m = IF_RE.match(line)
        if m:
            stack.append(m.group(1).strip())
            continue
        m = ELSEIF_RE.match(line)
        if m and stack:
            stack[-1] = m.group(1).strip()
            continue
        if ELSE_RE.match(line) and stack:
            stack[-1] = "NOT(" + stack[-1] + ")"
            continue
        if ENDIF_RE.match(line):
            if stack:
                stack.pop()
            continue

        if ADD_TEST_RE.match(line):
            buf, buf_conds = [line], list(stack)
            depth = line.count("(") - line.count(")")
            if depth <= 0:                      # whole call on one line
                m = NAME_RE.search(line)
                if m:
                    found.append((m.group(1), buf_conds))
                buf, buf_conds, depth = None, None, 0
    return found


def expected_set(entries, held):
    """Tests whose every enclosing condition is satisfied. Unknown condition -> abort."""
    unknown = sorted({c for _, conds in entries for c in conds
                      if c not in KNOWN_CONDITIONS})
    if unknown:
        raise SystemExit(
            "This CMakeLists gates tests on condition(s) this skill does not model:\n"
            + "".join(f"    {c}\n" for c in unknown)
            + "Refusing to guess - an unmodelled condition silently mis-tiers every test\n"
              "under it. Add it to KNOWN_CONDITIONS in check_ctest.py (and give it a flag)\n"
              "once you have decided what it depends on."
        )
    return sorted(n for n, conds in entries
                  if all(held.get(KNOWN_CONDITIONS[c], False) for c in conds))


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
    is pinned for its lifetime: exporting a different one changes nothing, and the
    top-level -DROCM_PATH never reaches these sub-projects at all.

    So exporting ROCM_PATH is necessary but NOT sufficient. Reuse build/ across two ROCm
    installs and the install check verifies one while the suite links the other.
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
    res = subprocess.run(["ctest", "-N"], cwd=build_dir, capture_output=True, text=True)
    return re.findall(r"^\s*Test\s+#\d+:\s+(\S+)\s*$", res.stdout, re.M)


def parse_results(log_text):
    """Pull pass/fail counts and failing names out of a ctest run log."""
    passed = failed = total = None
    m = re.search(r"(\d+)%\s+tests passed,\s+(\d+)\s+tests failed out of\s+(\d+)", log_text)
    if m:
        failed, total = int(m.group(2)), int(m.group(3))
        passed = total - failed
    failing = re.findall(
        r"^\s*\d+\s+-\s+(\S+)\s+\((?:Failed|Timeout|SEGFAULT|Subprocess aborted|Exception)[^)]*\)",
        log_text, re.M)
    return passed, failed, total, failing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--source-dir", required=True,
                    help="the dir holding the CMakeLists.txt that registers the tests")
    ap.add_argument("--suite", choices=["cpp_api", "pybind"], default="cpp_api")
    ap.add_argument("--backend", choices=["cpu", "hip"],
                    help="cpp_api only: the BACKEND the tests were configured with")
    ap.add_argument("--rocjpeg", action="store_true", help="rocjpeg_FOUND was true")
    ap.add_argument("--video-tests", action="store_true",
                    help="ROCAL_VIDEO_TESTS_ENABLED was true")
    ap.add_argument("--video-rocdecode", action="store_true",
                    help="ROCAL_VIDEO_ROCDECODE_AVAILABLE was true")
    ap.add_argument("--rocm-path",
                    help="required for cpp_api: the ROCM_PATH this run validates")
    ap.add_argument("--clean-stale-cache", action="store_true",
                    help="delete nested build dirs pinned to a different ROCM_PATH")
    ap.add_argument("--log", help="path to a ctest run log; if omitted, this script runs ctest")
    args = ap.parse_args()

    # ---- nested-cache staleness (before anything builds) --------------------
    # Must come first: ctest -N only lists, but the run below BUILDS each case in those
    # nested dirs, so a pinned ROCM_PATH has to be caught before anything is compiled
    # against the wrong install.
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
            print("  OLD install. Re-run with --clean-stale-cache, or delete them.")
            return 1

    # ---- expected set, derived ----------------------------------------------
    if args.suite == "cpp_api" and not args.backend:
        ap.error("cpp_api needs --backend cpu|hip")

    entries = registered_tests(os.path.join(args.source_dir, "CMakeLists.txt"))
    held = {
        "hip": args.backend == "hip",
        "rocjpeg": args.rocjpeg,
        "video_tests": args.video_tests,
        "video_rocdecode": args.video_rocdecode,
        # tests/pybind registers everything under one Python3_FOUND AND
        # ROCAL_PYBIND_SCRIPTS block; the gate for it not holding is "zero discovered",
        # handled below as a hard failure, so the expected set always assumes it held.
        "pybind_scripts": True,
    }
    expected = expected_set(entries, held)
    print(f"  source     : {args.source_dir}/CMakeLists.txt "
          f"({len(entries)} add_test, {len(expected)} expected here)")
    if args.suite == "cpp_api":
        print(f"  config     : backend={args.backend} rocjpeg={args.rocjpeg} "
              f"video_tests={args.video_tests} video_rocdecode={args.video_rocdecode}")

    discovered = sorted(set(discover(args.build_dir)))
    print(f"  discovered : {len(discovered)} test(s)")

    problems = []
    if not discovered:
        problems.append(
            "ZERO tests discovered. ctest reports 'No tests were found!!!' and still exits "
            "0, so this would otherwise read as a pass. For tests/pybind it means rocAL "
            "pybind is not installed at $ROCM_PATH/lib/amd/rocal."
        )
    missing = sorted(set(expected) - set(discovered))
    extra = sorted(set(discovered) - set(expected))
    if missing:
        problems.append("MISSING (silently not registered): " + ", ".join(missing))
    if extra:
        problems.append(
            "UNEXPECTED - registered but not predicted from CMakeLists under these flags. "
            "Check the flags match what configure actually found: " + ", ".join(extra))

    # ---- run results --------------------------------------------------------
    if args.log and os.path.isfile(args.log):
        log_text = open(args.log, errors="replace").read()
    else:
        res = subprocess.run(["ctest", "--output-on-failure"], cwd=args.build_dir,
                             capture_output=True, text=True)
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
        if total != len(discovered):
            problems.append(
                f"ran {total} but {len(discovered)} were registered - a filter or a crash "
                "cut the run short")

    if problems:
        print("\nCTEST GATE FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print("\nctest gate passed: the full expected set ran and every test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
