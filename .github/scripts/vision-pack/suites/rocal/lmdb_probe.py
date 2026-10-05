#!/usr/bin/env python3
# Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""LMDB probe (C2, M14, CHECK_LMDB_RETURN_STATUS) on private copies only.

    lmdb_probe.py all --conv DIR     (copies in $VP_ROCAL_LMDB, made by vp_copy_lmdb)

Per database (caffe-cls, caffe-det, caffe2-cls, caffe2-det), group ``lmdb``:
  bundled-open.<db>          the bundled liblmdb opens the dataset read-only (C2: MDB_INVALID)
  host-open.<db>             control: the host's LMDB 0.9 opens it (blocked without liblmdb.so.0)
  convert.<db>               dump with host LMDB 0.9, load with the bundled LMDB (new-format copy)
  rocal-read-converted.<db>  rocAL's reader on the converted copy (proves C2 is the file format)
  lock-unchanged.<db>        a rocAL read must not rewrite lock.mdb (M14)
and ``lmdb::error-message``: rocAL's error for the original copy must carry the real
mdb_env_open error; CHECK_LMDB_RETURN_STATUS evaluates its argument twice and reports
the second call's errno instead.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import os
import re
import sys
from pathlib import Path

from common import lmdb_root, py_child, record, result

DBS = {"caffe-cls": ("caffe", "classification", "caffe-lmdb"), "caffe-det": ("caffe", "detection", "caffe-lmdb-det"),
       "caffe2-cls": ("caffe2", "classification", "caffe2-lmdb"),
       "caffe2-det": ("caffe2", "detection", "caffe2-lmdb-det")}
MDB_RDONLY = 0x20000
MDB_NEXT = 8


class MDBVal(ctypes.Structure):
    _fields_ = [("mv_size", ctypes.c_size_t), ("mv_data", ctypes.c_void_p)]


def bundled_lib() -> str:
    return str(Path(os.environ["ROCM_PATH"]) / "lib" / "rocm_sysdeps" / "lib" / "liblmdb-rocm-vision.so.1")


def host_lib() -> str | None:
    name = ctypes.util.find_library("lmdb")
    return name


def load(path: str):
    lib = ctypes.CDLL(path)
    lib.mdb_version.restype = ctypes.c_char_p
    lib.mdb_strerror.restype = ctypes.c_char_p
    lib.mdb_env_open.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint, ctypes.c_int]
    lib.mdb_env_close.argtypes = [ctypes.c_void_p]
    lib.mdb_env_set_mapsize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.mdb_txn_begin.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
    lib.mdb_txn_commit.argtypes = [ctypes.c_void_p]
    lib.mdb_dbi_open.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint, ctypes.c_void_p]
    lib.mdb_cursor_open.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
    lib.mdb_cursor_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.mdb_put.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    lib.mdb_env_stat.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return lib


def open_env(lib, path: str, flags: int):
    env = ctypes.c_void_p()
    lib.mdb_env_create(ctypes.byref(env))
    rc = lib.mdb_env_open(env, path.encode(), flags, 0o664)
    return env, rc


def c_open(lib_path: str, db: str) -> None:
    lib = load(lib_path)
    ver = lib.mdb_version(None, None, None).decode()
    env, rc = open_env(lib, db, MDB_RDONLY)
    msg = f"{ver}: mdb_env_open(RDONLY) rc={rc} ({lib.mdb_strerror(rc).decode()})"
    lib.mdb_env_close(env)
    result("pass" if rc == 0 else "fail", msg + ("" if rc == 0 else " (C2)"))


def c_dump(lib_path: str, db: str, out: str) -> None:
    lib = load(lib_path)
    env, rc = open_env(lib, db, MDB_RDONLY)
    if rc:
        result("fail", f"open: {lib.mdb_strerror(rc).decode()}")
        return
    txn, dbi, cur = ctypes.c_void_p(), ctypes.c_uint(), ctypes.c_void_p()
    lib.mdb_txn_begin(env, None, MDB_RDONLY, ctypes.byref(txn))
    lib.mdb_dbi_open(txn, None, 0, ctypes.byref(dbi))
    lib.mdb_cursor_open(txn, dbi, ctypes.byref(cur))
    k, v, n = MDBVal(), MDBVal(), 0
    with open(out, "wb") as f:
        while lib.mdb_cursor_get(cur, ctypes.byref(k), ctypes.byref(v), MDB_NEXT) == 0:
            for val in (k, v):
                f.write(val.mv_size.to_bytes(8, "little"))
                f.write(ctypes.string_at(val.mv_data, val.mv_size))
            n += 1
    lib.mdb_env_close(env)
    result("pass" if n else "fail", f"dumped {n} records with {lib.mdb_version(None, None, None).decode()}")


def c_load(lib_path: str, inp: str, dst: str) -> None:
    lib = load(lib_path)
    Path(dst).mkdir(parents=True, exist_ok=True)
    env = ctypes.c_void_p()
    lib.mdb_env_create(ctypes.byref(env))
    lib.mdb_env_set_mapsize(env, 64 << 20)
    rc = lib.mdb_env_open(env, dst.encode(), 0, 0o664)
    if rc:
        result("fail", f"open for writing: {lib.mdb_strerror(rc).decode()}")
        return
    txn, dbi = ctypes.c_void_p(), ctypes.c_uint()
    lib.mdb_txn_begin(env, None, 0, ctypes.byref(txn))
    lib.mdb_dbi_open(txn, None, 0, ctypes.byref(dbi))
    data, pos, n = Path(inp).read_bytes(), 0, 0
    keep = []
    while pos < len(data):
        vals = []
        for _ in range(2):
            size = int.from_bytes(data[pos:pos + 8], "little")
            buf = ctypes.create_string_buffer(data[pos + 8:pos + 8 + size], size)
            keep.append(buf)
            vals.append(MDBVal(size, ctypes.cast(buf, ctypes.c_void_p)))
            pos += 8 + size
        rc = lib.mdb_put(txn, dbi, ctypes.byref(vals[0]), ctypes.byref(vals[1]), 0)
        if rc:
            result("fail", f"mdb_put: {lib.mdb_strerror(rc).decode()}")
            return
        n += 1
    rc = lib.mdb_txn_commit(txn)
    lib.mdb_env_close(env)
    result("pass" if rc == 0 else "fail", f"loaded {n} records with {lib.mdb_version(None, None, None).decode()}")


def c_rocal_error(kind: str, path: str) -> None:
    """Capture rocAL's LMDB error text for an unreadable (0.9-format) database."""
    import contextlib
    import io

    from readers import child as reader_child
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reader_child(kind, "cpu", path)
    text = buf.getvalue()
    print(text, flush=True)
    if '"pass"' in text:
        result("skip", "the reader opened the database, so there is no LMDB error to inspect")
    elif "MDB_INVALID" in text or "not an LMDB file" in text:
        result("pass", "the error names the real mdb_env_open failure")
    else:
        m = re.search(r"LMDB error.*?(mdb_env_open\S*?\)\s*:?\s*[^\"|]*)", text)
        seen = m.group(1).strip() if m else text.strip()[-300:]
        result("fail", f"the real error is MDB_INVALID (-30793), but rocAL reports '{seen[-200:]}' "
               "(CHECK_LMDB_RETURN_STATUS evaluates its argument twice)")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "absent"


def run_all(conv: Path) -> None:
    copies = lmdb_root()
    host = host_lib()
    for db, (fam, sub, kind) in DBS.items():
        src = copies / fam / sub
        py_child(f"lmdb::bundled-open.{db}", "lmdb_probe.py", ["child-open", bundled_lib(), str(src)])
        if not host:
            for name in ("host-open", "convert", "rocal-read-converted", "lock-unchanged"):
                record(f"lmdb::{name}.{db}", "blocked", "no host LMDB 0.9 (liblmdb.so.0) to read the original format")
            continue
        py_child(f"lmdb::host-open.{db}", "lmdb_probe.py", ["child-open", host, str(src)])
        dump = conv / f"{db}.rec"
        dst = conv / db
        r1 = py_child(f"lmdb::convert.{db}.dump", "lmdb_probe.py", ["child-dump", host, str(src), str(dump)])
        ok = r1["status"] == "pass" and py_child(
            f"lmdb::convert.{db}", "lmdb_probe.py", ["child-load", bundled_lib(), str(dump), str(dst)])["status"] == "pass"
        if not ok:
            for name in ("rocal-read-converted", "lock-unchanged"):
                record(f"lmdb::{name}.{db}", "blocked", "conversion to the bundled LMDB format failed")
            continue
        before = sha(dst / "lock.mdb")
        py_child(f"lmdb::rocal-read-converted.{db}", "readers.py", ["child", kind, "cpu", str(dst)], backend="CPU")
        after = sha(dst / "lock.mdb")
        record(f"lmdb::lock-unchanged.{db}", "pass" if before == after else "fail",
               "lock.mdb unchanged by a read" if before == after else
               f"a read-only rocAL reader rewrote lock.mdb ({before[:12]} -> {after[:12]}) (M14)")
    py_child("lmdb::error-message", "lmdb_probe.py", ["child-rocal-error", "caffe-lmdb", str(copies / "caffe" /
                                                                                          "classification")])


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1].startswith("child-"):
        fn = {"child-open": c_open, "child-dump": c_dump, "child-load": c_load, "child-rocal-error": c_rocal_error}
        fn[sys.argv[1]](*sys.argv[2:])
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all"])
    ap.add_argument("--conv", required=True)
    a = ap.parse_args()
    Path(a.conv).mkdir(parents=True, exist_ok=True)
    run_all(Path(a.conv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
