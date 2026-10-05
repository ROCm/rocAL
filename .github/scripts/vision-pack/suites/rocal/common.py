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

"""Shared helpers for the rocal suite's Python harnesses.

Every probe runs in its own child process (Pipeline.build() calls exit(0) on a failed
verify, M8, and several rocAL bugs crash the process) in a fresh working directory. The
child prints one ``RESULT {"status": ..., "message": ...}`` line; the parent records it,
or derives error/fail from the exit status when the child died before reporting.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(os.environ.get("VP_REPO", Path(__file__).resolve().parents[2])) / "build_tools" / "results"))
from emit import append_record  # noqa: E402

SUITE = "rocal"
HERE = Path(__file__).resolve().parent


def record(test_id: str, status: str, message: str = "", **kw) -> None:
    append_record(os.environ["VP_RESULTS"], SUITE, test_id, status, message=message, **kw)


def result(status: str, message: str = "", **extra) -> None:
    """Called in a child: report the verdict to the parent."""
    print("RESULT " + json.dumps({"status": status, "message": message, **extra}), flush=True)


def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in s.replace("::", "__"))[:180]


def rel(p: str) -> str:
    out = os.environ.get("VP_OUT", "")
    return p[len(out) + 1:] if out and p.startswith(out + "/") else p


def run_child(test_id: str, argv: list[str], timeout: int = 300, env: dict | None = None,
              backend: str = "", expect_hang: bool = False) -> dict:
    """Run argv in a fresh directory under $VP_WORK and record its RESULT line.

    With expect_hang, a timeout is the known-bug outcome and is recorded as error
    (hang), while a clean RESULT is still recorded as given.
    """
    work = Path(os.environ["VP_WORK"]) / "probes" / slug(test_id)
    work.mkdir(parents=True, exist_ok=True)
    log = Path(os.environ["VP_OUT"]) / "logs" / f"{slug(test_id)}.log"
    full_env = dict(os.environ)
    full_env.update(env or {})
    t0 = time.monotonic()
    timed_out = False
    with open(log, "w") as fh:
        fh.write(f"### {test_id}\n### cwd: {work}\n### cmd: {' '.join(argv)}\n")
        fh.flush()
        try:
            p = subprocess.run(argv, cwd=work, env=full_env, stdout=fh, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = 124, True
    dur = time.monotonic() - t0
    text = log.read_text(errors="replace")
    res = None
    for line in text.splitlines():
        if line.startswith("RESULT "):
            try:
                res = json.loads(line[7:])
            except json.JSONDecodeError:
                pass
    tail = " | ".join([ln for ln in text.splitlines()[3:] if ln.strip() and not ln.startswith("OK:")][-4:])[:1500]
    if res is not None and not timed_out:
        status, msg = res["status"], res.get("message", "")
        if rc != 0 and status == "pass":
            status, msg = "fail", f"reported pass but exited {rc}; {msg}"
    elif timed_out:
        status = "error"
        msg = f"hung: no result after {timeout}s" + (" (the known hang)" if expect_hang else "") + f"; {tail}"
    elif rc < 0 or rc > 128:
        sig = -rc if rc < 0 else rc - 128
        status, msg = "error", f"killed by signal {sig}; {tail}"
    else:
        status, msg = "fail", f"exit {rc} without a RESULT line; {tail}"
    record(test_id, status, msg, duration_s=dur, log=rel(str(log)), backend=backend,
           repro=f"(cd {work} && {' '.join(argv)})")
    return {"status": status, "message": msg, "rc": rc, "result": res}


def py_child(test_id: str, script: str, args: list[str], **kw) -> dict:
    return run_child(test_id, [os.environ.get("VP_PY", sys.executable), str(HERE / script), *args], **kw)


def data_root() -> Path:
    return Path(os.environ["VP_DATA"]) / "rocal_data"


def tiny_dataset() -> Path:
    return Path(os.environ["ROCM_PATH"]) / "share" / "rocal" / "test" / "data" / "images" / "AMD-tinyDataSet"


def lmdb_root() -> Path:
    return Path(os.environ["VP_ROCAL_LMDB"])


def fetch(pipe, index: int = 0):
    """Run one iteration and return output tensor `index` as a host numpy array."""
    import numpy as np
    if pipe.rocal_run() != 0:
        raise RuntimeError("rocal_run() returned non-zero (end of data or runtime error)")
    tl = pipe.get_output_tensors()[index]
    out = np.zeros(tl.dimensions(), dtype=tl.dtype())
    tl.copy_data(out)
    return out
