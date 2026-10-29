#!/usr/bin/env python3
"""Verify that `make install` actually landed the build tree that was just built.

Why this exists
---------------
`make install` rewrites RUNPATH in place, so the installed librocal.so differs from
build/lib/librocal.so byte-for-byte *at identical size and mtime*. A whole-file checksum
therefore always mismatches and tells you nothing, while size/mtime always match and also
tell you nothing. Comparing the `.text` section is the check that actually discriminates:
it is invariant under the install-time RUNPATH edit but changes whenever the code changes.

A stale install is not a loud failure. ctest builds every test as a standalone sub-project
that resolves rocAL via find_library(... HINTS ${ROCM_PATH}/lib), so without a fresh install
the whole suite silently re-validates the *previous* build and reports green.

Usage:
    verify_install.py --build-dir build --rocm-path "$ROCM_PATH" --repo-root .

Exit 0 = install matches the build tree. Exit 1 = mismatch (every offender named).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from glob import glob

HEADERS = [
    "rocal_api.h",
    "rocal_api_augmentation.h",
    "rocal_api_data_loaders.h",
    "rocal_api_data_transfer.h",
    "rocal_api_info.h",
    "rocal_api_meta_data.h",
    "rocal_api_parameters.h",
    "rocal_api_tensor.h",
    "rocal_api_types.h",
]


def text_md5(so_path):
    """md5 of the .text section only - invariant under install-time RUNPATH rewriting."""
    try:
        out = subprocess.run(
            ["objcopy", "-O", "binary", "--only-section=.text", so_path, "/dev/stdout"],
            capture_output=True, check=True,
        ).stdout
    except FileNotFoundError:
        return None, "objcopy not found (binutils missing)"
    except subprocess.CalledProcessError as exc:
        return None, f"objcopy failed: {exc.stderr.decode(errors='replace').strip()}"
    if not out:
        return None, "empty .text section"
    return hashlib.md5(out).hexdigest(), None


def real_files(paths):
    """Drop symlinks - librocal.so and librocal.so.2 both point at librocal.so.2.5.0."""
    return [p for p in paths if os.path.isfile(p) and not os.path.islink(p)]


def pair_by_name(built_paths, installed_paths):
    """Pair build-tree and installed libraries by basename.

    Deliberately not "the biggest" or "the most recent". Both are guesses, and both can
    pair the wrong two files:

      * A ROCm tree routinely carries rocal_pybind cpython-311, -312 and -313 side by
        side, so size would happily compare the 3.12 build against the 3.13 install.
      * mtime is worse than useless here: `make install` preserves it, which is precisely
        why mtime cannot detect install skew in the first place.

    Basename pairing removes the guess. A built file with no installed counterpart is
    reported as "install did not land" - which is the correct verdict when the version
    was bumped and the new library was never installed.
    """
    installed = {os.path.basename(p): p for p in real_files(installed_paths)}
    return [(os.path.basename(b), b, installed.get(os.path.basename(b)))
            for b in real_files(built_paths)]


def check_pair(label, built, installed, findings):
    if built is None:
        findings.append(("FAIL", label, "not found in the build tree - did the build run?"))
        return
    if installed is None:
        findings.append(("FAIL", label, "not found under ROCM_PATH - install did not land"))
        return
    b, berr = text_md5(built)
    i, ierr = text_md5(installed)
    if berr or ierr:
        findings.append(("FAIL", label, berr or ierr))
        return
    if b != i:
        findings.append((
            "FAIL", label,
            f".text differs - installed copy is STALE\n"
            f"      built:     {built}  ({b})\n"
            f"      installed: {installed}  ({i})",
        ))
    else:
        findings.append(("OK", label, f".text matches ({b[:12]}...)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--rocm-path", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--json", action="store_true", help="emit a machine-readable summary")
    args = ap.parse_args()

    findings = []

    # 1. librocal.so
    lib_pairs = pair_by_name(
        glob(os.path.join(args.build_dir, "lib", "librocal.so*")),
        glob(os.path.join(args.rocm_path, "lib", "librocal.so*")),
    )
    if not lib_pairs:
        findings.append((
            "FAIL", "librocal.so", "not found in the build tree - did the build run?",
        ))
    for name, built, installed in lib_pairs:
        check_pair(name, built, installed, findings)

    # 2. rocal_pybind extension module(s)
    pb_pairs = pair_by_name(
        glob(os.path.join(args.build_dir, "**", "rocal_pybind*.so"), recursive=True),
        glob(os.path.join(args.rocm_path, "lib", "rocal_pybind*.so")),
    )
    if not pb_pairs:
        findings.append((
            "SKIP", "rocal_pybind",
            "not built (pybind disabled) - phases 4 and 5 will be skipped",
        ))
    for name, built, installed in pb_pairs:
        check_pair(name, built, installed, findings)

    # 3. public headers
    src_inc = os.path.join(args.repo_root, "rocAL", "include", "api")
    dst_inc = os.path.join(args.rocm_path, "include", "rocal")
    missing, differing = [], []
    for h in HEADERS:
        s, d = os.path.join(src_inc, h), os.path.join(dst_inc, h)
        if not os.path.isfile(d):
            missing.append(h)
        elif not os.path.isfile(s):
            continue  # header removed from the source tree; not an install problem
        elif open(s, "rb").read() != open(d, "rb").read():
            differing.append(h)
    if missing or differing:
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if differing:
            detail.append("stale: " + ", ".join(differing))
        findings.append(("FAIL", "headers", "; ".join(detail)))
    else:
        findings.append(("OK", "headers", f"all {len(HEADERS)} match {dst_inc}"))

    # 4. test data payload that every ctest case passes as argv
    data_dir = os.path.join(args.rocm_path, "share", "rocal", "test", "data")
    if os.path.isdir(data_dir):
        findings.append(("OK", "test data", data_dir))
    else:
        findings.append(("FAIL", "test data", f"{data_dir} absent - ctest cases will fail on missing argv paths"))

    # 5. pybind scripts dir - tests/pybind/CMakeLists.txt gates every add_test on this
    pb_scripts = os.path.join(args.rocm_path, "lib", "amd", "rocal")
    if os.path.isdir(pb_scripts):
        findings.append(("OK", "pybind scripts", pb_scripts))
    else:
        findings.append((
            "WARN", "pybind scripts",
            f"{pb_scripts} absent - tests/pybind will discover 0 tests and still exit 0",
        ))

    width = max(len(label) for _, label, _ in findings)
    for status, label, detail in findings:
        print(f"  [{status:4}] {label:<{width}}  {detail}")

    failed = [f for f in findings if f[0] == "FAIL"]
    if args.json:
        print("JSON " + json.dumps({
            "ok": not failed,
            "findings": [{"status": s, "item": l, "detail": d} for s, l, d in findings],
        }))

    if failed:
        print(f"\nINSTALL VERIFICATION FAILED ({len(failed)} problem(s)).")
        print("The installed tree does not match what was just built. Re-run:")
        print(f"    cd {args.build_dir} && sudo make install")
        print("Do NOT continue - ctest would re-validate the previous build and report green.")
        return 1
    print("\nInstall verified: the installed tree matches this build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
