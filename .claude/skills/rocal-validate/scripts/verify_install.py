#!/usr/bin/env python3
"""Verify that `make install` actually landed the build tree that was just built.

Why this exists
---------------
`make install` rewrites RUNPATH in place, so the installed librocal.so differs from
build/lib/librocal.so byte-for-byte *at identical size and mtime*. A whole-file checksum
therefore always mismatches and tells you nothing, while size/mtime always match and also
tell you nothing.

A stale install is not a loud failure. ctest builds every test as a standalone sub-project
that resolves rocAL via find_library(... HINTS ${ROCM_PATH}/lib), so without a fresh
install the whole suite silently re-validates the *previous* build and reports green.

What is compared
----------------
Every ELF section's name, size and content, plus the dynamic entries - with only the
install-volatile parts normalised away. Hashing `.text` alone is not enough: it covers
host instructions only, so a change confined to `.rodata` (lookup tables) or `.hip_fatbin`
(embedded GPU device code) would slip through. Measured on a real librocal.so.2.5.0, a
RUNPATH rewrite changes exactly one section - `.dynstr` - while `.rodata`, `.hip_fatbin`,
`.data` and `.data.rel.ro` are untouched. So everything except the RUNPATH carrier can be
compared directly.

The installed Python package is compared too. A change confined to
rocAL_pybind/amd/rocal/fn.py leaves every .so and header identical, and the python_api
suite imports the *installed* package - so without this check it would validate the
previous implementation.

Usage:
    verify_install.py --build-dir build --rocm-path "$ROCM_PATH" --repo-root .

Exit 0 = install matches the build tree. Exit 1 = mismatch (every offender named).
"""

import argparse
import hashlib
import json
import os
import re
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

# Sections that `make install` legitimately rewrites when it fixes up RUNPATH. .dynstr is
# the string table the RUNPATH lives in; .dynamic holds the DT_RUNPATH entry pointing into
# it. Both are excluded from raw comparison and checked via normalised `readelf -d`
# instead, so a genuine change to NEEDED libraries is still caught.
INSTALL_VOLATILE = {".dynstr", ".dynamic"}

RUNPATH_TAG = re.compile(r"\((RUNPATH|RPATH)\)")
DYN_ROW = re.compile(r"^\s*0x[0-9a-fA-F]+\s+\(")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def elf_sections(path):
    """-> {name: size} for every section, or None if the file is not readable ELF."""
    res = run(["readelf", "-S", "--wide", path])
    if res.returncode != 0:
        return None
    out = {}
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line.startswith("[") or "]" not in line:
            continue
        rest = line.split("]", 1)[1].strip().split()
        # name type addr off size ...
        if len(rest) >= 5 and rest[0].startswith("."):
            try:
                out[rest[0]] = int(rest[4], 16)
            except ValueError:
                continue
    return out


def section_md5(path, name):
    res = subprocess.run(
        ["objcopy", "-O", "binary", f"--only-section={name}", path, "/dev/stdout"],
        capture_output=True)
    if res.returncode != 0:
        return None
    return hashlib.md5(res.stdout).hexdigest()


def dynamic_entries(path):
    """Dynamic-entry rows only, minus RPATH/RUNPATH.

    The `Dynamic section ... contains N entries:` header is excluded on purpose: the top
    level sets CMAKE_SKIP_INSTALL_RPATH, so install drops the RPATH entry entirely and N
    changes, which would flag a legitimate install as stale.
    """
    res = run(["readelf", "-d", "--wide", path])
    if res.returncode != 0:
        return None
    return [l.strip() for l in res.stdout.splitlines()
            if DYN_ROW.match(l) and not RUNPATH_TAG.search(l)]


def compare_binaries(label, built, installed, findings):
    if built is None:
        findings.append(("FAIL", label, "not found in the build tree - did the build run?"))
        return
    if installed is None:
        findings.append((
            "FAIL", label,
            "no installed counterpart of the same name - install did not land "
            "(or the version was bumped and never installed)",
        ))
        return

    sa, sb = elf_sections(built), elf_sections(installed)
    if sa is None or sb is None:
        findings.append(("FAIL", label, "readelf could not parse one of the files"))
        return

    problems = []

    only_built = sorted(set(sa) - set(sb))
    only_inst = sorted(set(sb) - set(sa))
    if only_built or only_inst:
        bits = []
        if only_built:
            bits.append("missing from installed: " + ", ".join(only_built))
        if only_inst:
            bits.append("extra in installed: " + ", ".join(only_inst))
        problems.append("section layout differs (" + "; ".join(bits) + ")")

    differing = []
    for name in sorted(set(sa) & set(sb)):
        if name in INSTALL_VOLATILE:
            continue
        if sa[name] != sb[name]:
            differing.append(f"{name} (size {sa[name]} vs {sb[name]})")
            continue
        ha, hb = section_md5(built, name), section_md5(installed, name)
        if ha is None or hb is None:
            problems.append(f"objcopy failed on {name}")
        elif ha != hb:
            differing.append(name)

    if differing:
        problems.append("content differs in: " + ", ".join(differing))

    da, db = dynamic_entries(built), dynamic_entries(installed)
    if da is not None and db is not None and da != db:
        problems.append(
            "dynamic entries differ beyond RUNPATH (a NEEDED library changed)")

    if problems:
        findings.append((
            "FAIL", label,
            "installed copy is STALE - " + "; ".join(problems) + "\n"
            f"      built:     {built}\n"
            f"      installed: {installed}",
        ))
    else:
        n = len([k for k in sa if k not in INSTALL_VOLATILE])
        findings.append(("OK", label, f"all {n} comparable sections match"))


def check_aliases(libdir, pattern, verified, findings):
    """Require every symlink alias to resolve to content already verified as matching."""
    aliases = [p for p in glob(os.path.join(libdir, pattern)) if os.path.islink(p)]
    if not aliases:
        findings.append(("WARN", "so aliases",
                         f"no librocal symlinks in {libdir} - find_library() needs librocal.so"))
        return
    problems = []
    for a in sorted(aliases):
        target = os.path.realpath(a)
        if not os.path.exists(target):
            problems.append(f"{os.path.basename(a)} -> {os.readlink(a)} (DANGLING)")
        elif verified and target not in verified:
            problems.append(
                f"{os.path.basename(a)} -> {os.path.basename(target)} "
                "(resolves to a library that is not the one just verified)")
    if problems:
        findings.append(("FAIL", "so aliases", "; ".join(problems)))
    else:
        findings.append(("OK", "so aliases",
                         f"{len(aliases)} alias(es) resolve to verified content"))


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
    """
    installed = {os.path.basename(p): p for p in real_files(installed_paths)}
    return [(os.path.basename(b), b, installed.get(os.path.basename(b)))
            for b in real_files(built_paths)]


def compare_tree(label, src_dir, dst_dir, pattern, findings, missing_is_fail=True):
    """Byte-compare every file matching `pattern` under src_dir against dst_dir."""
    if not os.path.isdir(src_dir):
        findings.append(("SKIP", label, f"{src_dir} absent in the checkout"))
        return
    if not os.path.isdir(dst_dir):
        findings.append((
            "FAIL" if missing_is_fail else "WARN", label,
            f"{dst_dir} absent - nothing was installed",
        ))
        return

    missing, differing, n = [], [], 0
    for src in sorted(glob(os.path.join(src_dir, pattern), recursive=True)):
        if not os.path.isfile(src):
            continue
        rel = os.path.relpath(src, src_dir)
        dst = os.path.join(dst_dir, rel)
        n += 1
        if not os.path.isfile(dst):
            missing.append(rel)
        elif open(src, "rb").read() != open(dst, "rb").read():
            differing.append(rel)

    if missing or differing:
        bits = []
        if missing:
            bits.append(f"missing ({len(missing)}): " + ", ".join(missing[:10]))
        if differing:
            bits.append(f"STALE ({len(differing)}): " + ", ".join(differing[:10]))
        findings.append(("FAIL", label, "; ".join(bits)))
    else:
        findings.append(("OK", label, f"all {n} file(s) match {dst_dir}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--rocm-path", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--json", action="store_true", help="emit a machine-readable summary")
    args = ap.parse_args()

    findings = []

    # 1. librocal.so - the build tree holds one real file plus two symlinks
    lib_pairs = pair_by_name(
        glob(os.path.join(args.build_dir, "lib", "librocal.so*")),
        glob(os.path.join(args.rocm_path, "lib", "librocal.so*")),
    )
    if not lib_pairs:
        findings.append((
            "FAIL", "librocal.so", "not found in the build tree - did the build run?",
        ))
    verified = set()
    for name, built, installed in lib_pairs:
        before = len([f for f in findings if f[0] == "FAIL"])
        compare_binaries(name, built, installed, findings)
        if len([f for f in findings if f[0] == "FAIL"]) == before and installed:
            verified.add(os.path.realpath(installed))

    # The aliases are what tests actually link and load: find_library() resolves
    # librocal.so, and the loader follows SONAME librocal.so.2. A partial install can
    # leave either dangling or pointing at an older release while the versioned file is
    # correct, so resolve each one and require it to land on verified content.
    check_aliases(os.path.join(args.rocm_path, "lib"), "librocal.so*", verified, findings)

    # 2. rocal_pybind extension module(s), paired by basename
    pb_pairs = pair_by_name(
        glob(os.path.join(args.build_dir, "**", "rocal_pybind*.so"), recursive=True),
        glob(os.path.join(args.rocm_path, "lib", "rocal_pybind*.so")),
    )
    pybind_built = bool(pb_pairs)
    if not pb_pairs:
        findings.append((
            "SKIP", "rocal_pybind",
            "not built (pybind disabled) - phases 4 and 5 will be skipped",
        ))
    for name, built, installed in pb_pairs:
        compare_binaries(name, built, installed, findings)

    # 3. public headers
    compare_tree(
        "headers",
        os.path.join(args.repo_root, "rocAL", "include", "api"),
        os.path.join(args.rocm_path, "include", "rocal"),
        "*.h", findings)

    # 4. the installed Python package. rocAL_pybind/amd/ -> $ROCM_PATH/lib/amd/, and the
    #    python_api suite imports THAT, not the checkout. A change confined to fn.py
    #    leaves every binary identical, so this is the only check that catches it.
    if pybind_built:
        compare_tree(
            "pybind python pkg",
            os.path.join(args.repo_root, "rocAL_pybind", "amd"),
            os.path.join(args.rocm_path, "lib", "amd"),
            "**/*.py", findings)

    # 5. test data payload that every ctest case passes as argv
    data_dir = os.path.join(args.rocm_path, "share", "rocal", "test", "data")
    if os.path.isdir(data_dir):
        findings.append(("OK", "test data", data_dir))
    else:
        findings.append((
            "FAIL", "test data",
            f"{data_dir} absent - ctest cases will fail on missing argv paths",
        ))

    # 6. pybind scripts dir - tests/pybind/CMakeLists.txt gates every add_test on this
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
