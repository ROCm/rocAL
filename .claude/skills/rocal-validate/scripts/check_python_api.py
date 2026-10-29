#!/usr/bin/env python3
"""Gate the tests/python_api golden-image suite by parsing its summary.

Why this exists
---------------
Nothing under tests/python_api sets a meaningful exit code:

  * unit_tests.sh:4-8 guards a missing ROCAL_DATA_PATH with a bare `exit`, which exits 0.
  * image_comparison.py never calls sys.exit() on failure - it logs and returns.

So `bash unit_tests.sh 0 && echo ok` prints ok whether 0 or 139 cases failed. The only
signal is the summary text, and it has two traps:

  1. image_comparison.py:104 iterates over the *rocAL output* directory, not the goldens.
     A case that never produced an output file is not in that listing at all, so it is
     never counted - it vanishes from the denominator instead of failing. A shrunken
     total is therefore a REGRESSION, not a rounding difference.
  2. passed + failed can be < total: entries that hit the "Skipping the testcase as file
     not found" branch are counted in the denominator but in neither bucket.

The gate is on the failing *set*, not the count. 19 failures that are a different 19 than
the baseline means something broke and something else got fixed - that must not read green.

Usage:
    check_python_api.py --log <unit_tests output> --backend host --baselines baselines.json
                        [--output-dir <rocal_python_unittest_output_folder_*>]
                        [--golden-dir $ROCAL_DATA_PATH/rocal_data/GoldenOutputsTensor]

Exit 0 = matches baseline exactly. Exit 1 = regression. Exit 0 with notes = improvement.
"""

import argparse
import json
import os
import re
import sys

SUFFIX = re.compile(r"_(host|hip)$")


def norm(name):
    """'/path/Hue_rgb_host.png' -> 'Hue_rgb'  (drop dir, extension, backend suffix)."""
    name = os.path.basename(name.strip())
    if name.endswith(".png"):
        name = name[: -len(".png")]
    return SUFFIX.sub("", name)


def parse(log_text):
    passed = failed = total = None
    m = re.search(r"Total case passed -->\s*(\d+)\s*/\s*(\d+)", log_text)
    if m:
        passed, total = int(m.group(1)), int(m.group(2))
    m = re.search(r"Total case failed -->\s*(\d+)\s*/\s*(\d+)", log_text)
    if m:
        failed, total = int(m.group(1)), int(m.group(2))

    failing = []
    m = re.search(r"Failing cases:\s*(.+)", log_text)
    if m:
        failing = [norm(x) for x in m.group(1).split(",") if x.strip()]

    skipped = re.findall(r"Skipping the testcase as file not found\s+(\S+)", log_text)
    no_golden = re.findall(r"File not found in ref_output_folder\s+(\S+)", log_text)
    return passed, failed, total, sorted(set(failing)), skipped, no_golden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--backend", required=True, choices=["host", "hip"])
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--output-dir", help="the run's rocal_python_unittest_output_folder_*")
    ap.add_argument("--golden-dir", help="$ROCAL_DATA_PATH/rocal_data/GoldenOutputsTensor")
    ap.add_argument("--update", action="store_true",
                    help="rewrite this backend's baseline from the observed run")
    args = ap.parse_args()

    with open(args.baselines) as fh:
        base = json.load(fh)
    bl = base["python_api"][args.backend]

    log_text = open(args.log, errors="replace").read()
    passed, failed, total, failing, skipped, no_golden = parse(log_text)

    if total is None:
        print("PYTHON_API GATE FAILED: no 'Total case passed/failed' line in the log.")
        print("  The suite did not reach image_comparison.py. Common cause: ROCAL_DATA_PATH")
        print("  unset - unit_tests.sh exits 0 after printing 'Need to export ROCAL_DATA_PATH'.")
        return 1

    print(f"  backend    : {args.backend}")
    print(f"  observed   : passed {passed}, failed {failed}, total {total}")
    print(f"  baseline   : passed {bl['passed']}, failed {bl['failed']}, total {bl['total']}")

    problems, notes = [], []

    # --- denominator ---------------------------------------------------------
    if total != bl["total"]:
        delta = total - bl["total"]
        problems.append(
            f"TOTAL CHANGED: {total} vs baseline {bl['total']} ({delta:+d}). "
            "image_comparison.py counts only cases that produced an output file, so a smaller "
            "total means cases silently stopped running - treat as a regression, not noise."
        )

    # --- unaccounted cases ---------------------------------------------------
    if passed is not None and failed is not None and passed + failed != total:
        gap = total - (passed + failed)
        if gap > 0:
            problems.append(
                f"{gap} case(s) counted in the denominator but neither passed nor failed "
                "(the 'Skipping the testcase' branch)."
            )
        else:
            problems.append(
                f"passed + failed ({passed + failed}) EXCEEDS the reported total ({total}) "
                f"by {-gap}. The summary is internally inconsistent - treat the log as "
                "untrustworthy and re-run rather than interpreting these numbers."
            )
    for s in skipped:
        notes.append(f"skipped (no output file): {s}")
    for g in no_golden:
        notes.append(f"no golden in reference folder: {g}")

    # --- parse sanity --------------------------------------------------------
    # The names on the 'Failing cases:' line must account for the reported failed count.
    # If they do not, the line was truncated or the regex missed something, and every
    # set-based conclusion below would be drawn from bad input.
    if failed is not None and failing and len(failing) != failed:
        problems.append(
            f"parse inconsistency: {len(failing)} name(s) on the 'Failing cases:' line but "
            f"the summary reports {failed} failure(s). Do not trust the set comparison - "
            "check the log by hand."
        )

    # --- failing set ---------------------------------------------------------
    pinned = bl.get("failing")
    if pinned is None:
        notes.append(
            f"baseline failing set for '{args.backend}' is UNPINNED "
            f"({bl.get('failing_note', '')}). Observed set recorded below; re-run with "
            "--update to pin it."
        )
        print(f"  observed failing set ({len(failing)}): " + ", ".join(failing))
        if failed != bl["failed"]:
            problems.append(f"failed count {failed} vs baseline {bl['failed']}")
    else:
        pinned_set = set(pinned)
        observed = set(failing)
        new_failures = sorted(observed - pinned_set)
        now_passing = sorted(pinned_set - observed)

        if new_failures:
            problems.append("NEW FAILURES (not in baseline):\n      - " +
                            "\n      - ".join(new_failures))
        if now_passing:
            # Only call it an improvement if the denominator held. When the total shrank,
            # a name leaving the failing list usually means the case stopped running -
            # reporting that as "improved" would invert the truth.
            if total == bl["total"]:
                notes.append("IMPROVED - these were failing and now pass:\n      - " +
                             "\n      - ".join(now_passing))
            else:
                notes.append(
                    f"{len(now_passing)} baseline failure(s) are absent from the failing "
                    "list, but the total also changed, so they most likely STOPPED RUNNING "
                    "rather than started passing. Do NOT read these as fixes:\n      - " +
                    "\n      - ".join(now_passing))
        if not new_failures and not now_passing:
            print(f"  failing set: identical to baseline ({len(pinned_set)} known failures)")

    # --- cross-check against the filesystem ----------------------------------
    if args.output_dir and args.golden_dir and os.path.isdir(args.output_dir) \
            and os.path.isdir(args.golden_dir):
        produced = {norm(f) for f in os.listdir(args.output_dir)}
        goldens = {norm(f) for f in os.listdir(args.golden_dir)}
        never_ran = sorted(goldens - produced)
        if never_ran:
            notes.append(
                f"{len(never_ran)} golden(s) have no corresponding output in this run "
                f"(never entered the denominator): " + ", ".join(never_ran[:20]) +
                (" ..." if len(never_ran) > 20 else "")
            )

    # --- update --------------------------------------------------------------
    if args.update:
        bl.update({"total": total, "passed": passed, "failed": failed,
                   "failing": failing})
        bl.pop("failing_note", None)
        with open(args.baselines, "w") as fh:
            json.dump(base, fh, indent=2)
            fh.write("\n")
        print(f"\nbaselines.json updated for '{args.backend}'. Commit this deliberately.")
        return 0

    if notes:
        print("\nNotes:")
        for n in notes:
            print("  * " + n)

    if problems:
        print("\nPYTHON_API GATE FAILED:")
        for p in problems:
            print("  - " + p)
        return 1

    print(f"\npython_api gate passed for '{args.backend}'.")
    print("  CAVEAT: the baseline failures above are KNOWN but NOT root-caused. They are")
    print("  recorded as known, not as acceptable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
