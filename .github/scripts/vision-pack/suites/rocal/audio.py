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

"""rocAL audio QA (H5).

    audio.py python          audio_unit_test.py in QA mode, one process per case, CPU and --rocal-gpu
                             (audio.<CPU|GPU>::<case>), plus one upstream-style run of all cases
                             (audio.<CPU|GPU>::qa-run). Needs torch and matplotlib.
    audio.py cpp --bin B     the C++ audio_tests binary per case (audio-cpp.<CPU|GPU>::<case>),
                             with a trailing-slash ROCAL_DATA_PATH as the binary requires.
    audio.py nontorch        torch-free Python check of the decoder and pre-emphasis against the
                             reference outputs (audio-nontorch.<CPU|GPU>::<case>).

audio_unit_test.py always exits 0, and Pipeline.build() exits 0 on a failed verify (M8),
so verdicts come from its PASSED!/FAILED! lines and final lists, never the exit status.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

from common import py_child, record, result, run_child

CASES = {0: "audio_decoder", 1: "preemphasis_filter", 2: "spectrogram", 3: "downmix", 4: "to_decibels",
         5: "resample", 6: "tensor_add_tensor", 7: "tensor_mul_scalar", 8: "non_silent_region", 9: "slice",
         10: "mel_filter_bank", 11: "normalize"}
DEVS = (("CPU", []), ("GPU", ["--rocal-gpu"]))


def upstream_script() -> str:
    return str(Path(os.environ["ROCM_PATH"]) / "share" / "rocal" / "test" / "python_api" / "audio_unit_test.py")


def parse_lists(text: str) -> tuple[list | None, list | None]:
    lists = {}
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"Number of (PASSED|FAILED) tests:", ln)
        if m and i + 1 < len(lines):
            try:
                lists[m.group(1)] = ast.literal_eval(lines[i + 1].strip())
            except (ValueError, SyntaxError):
                pass
    return lists.get("PASSED"), lists.get("FAILED")


def c_py_case(case: str, *flags: str) -> None:
    name = CASES[int(case)]
    p = subprocess.run([sys.executable, upstream_script(), "--test_case", case, "--qa_mode", "1", "--no-display",
                        *flags], capture_output=True, text=True, timeout=600)
    text = p.stdout + p.stderr
    print(text[-20000:], flush=True)
    passed, failed = parse_lists(text)
    if p.returncode < 0:
        result("error", f"killed by signal {-p.returncode}")
    elif passed is not None and name in passed:
        result("pass", "QA PASSED")
    elif failed is not None and name in failed:
        result("fail", "QA FAILED: output differs from the reference (H5)")
    elif "Verify graph failed" in text:
        result("fail", f"rocalVerify failed and the script exited {p.returncode} without a verdict (H5, M8); "
               + " | ".join(ln for ln in text.splitlines() if "[ERR]" in ln)[:500])
    else:
        result("fail", f"no QA verdict for {name} (exit {p.returncode})")


def c_py_run(*flags: str) -> None:
    p = subprocess.run([sys.executable, upstream_script(), "--qa_mode", "1", "--no-display", *flags],
                       capture_output=True, text=True, timeout=1800)
    text = p.stdout + p.stderr
    print(text[-20000:], flush=True)
    passed, failed = parse_lists(text)
    if passed is None:
        result("fail", f"exit {p.returncode} without the PASSED/FAILED lists: the run stopped at the first failed "
               "build (Pipeline.build() exit(0), M8), so a green exit status hides every audio failure (H5)")
    elif failed:
        result("fail", f"{len(passed)} passed, {len(failed)} failed: {failed} (H5); exit status {p.returncode}")
    else:
        result("pass", f"{len(passed)} passed")


def c_cpp(binary: str, case: str, gpu: str) -> None:
    data = os.environ["VP_DATA"].rstrip("/") + "/"
    src = data + ("rocal_data/multi_channel_wav" if case == "3" else "rocal_data/audio")
    env = dict(os.environ, ROCAL_DATA_PATH=data)
    p = subprocess.run([binary, src, case, "1" if case == "3" else "0", gpu, "1"], capture_output=True, text=True,
                       timeout=300, env=env)
    text = p.stdout + p.stderr
    print(text[-20000:], flush=True)
    errs = " | ".join(ln.strip() for ln in text.splitlines() if "[ERR]" in ln)[:500]
    if "PASSED!" in text and p.returncode == 0:
        result("pass", "PASSED")
    elif "FAILED!" in text:
        result("fail", f"FAILED: output differs from the reference (H5) {errs}")
    elif p.returncode < 0:
        result("error", f"crashed with signal {-p.returncode} after: {errs or 'no error message'}")
    else:
        result("fail", f"exit {p.returncode} without a verdict: {errs}")


def c_nontorch(case: str, dev: str) -> None:
    """Decode (and optionally pre-emphasis) without torch; compare with the reference output."""
    import ctypes

    import amd.rocal.fn as fn
    import numpy as np
    from amd.rocal.pipeline import Pipeline
    from probes import safe_build
    data = Path(os.environ["VP_DATA"]) / "rocal_data"
    ref = np.fromfile(data / "GoldenOutputsTensor" / "reference_outputs_audio" / f"{case}_output.bin",
                      dtype=np.float32)
    root, flist = str(data / "audio") + "/", str(data / "audio" / "wav_file_list.txt")
    pipe = Pipeline(batch_size=1, num_threads=1, device_id=0, seed=0, rocal_cpu=(dev == "CPU"))
    with pipe:
        audio, _ = fn.readers.file(file_root=root, file_list=flist)
        dec = fn.decoders.audio(audio, file_root=root, file_list_path=flist, downmix=False, shard_id=0,
                                num_shards=1, stick_to_shard=False)
        pipe.set_outputs(dec if case == "audio_decoder" else fn.preemphasis_filter(dec))
    if not safe_build(pipe):
        return
    seen = []
    for _ in range(10):
        if pipe.rocal_run() != 0:
            break
        tl = pipe.get_output_tensors()[0]
        nroi = tl.roi_dims_size() * 2
        roi = np.zeros(nroi, dtype=np.int32)
        tl.copy_roi(roi)
        frames, chans = int(roi[nroi - 2]), int(roi[nroi - 1])
        out = np.zeros((1, frames, chans), dtype=np.float32)
        tl.copy_data(ctypes.c_void_p(out.ctypes.data), 0, 0, chans, frames)
        seen.append(frames * chans)
        if frames * chans == len(ref):
            err = float(np.abs(out.ravel() - ref).max())
            result("pass" if err < 1e-5 else "fail", f"max abs error {err:.3e} over {len(ref)} samples")
            return
    result("fail", f"no sample matched the reference length {len(ref)}; lengths seen {seen}")


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "child-py-case":
        c_py_case(*sys.argv[2:])
    elif cmd == "child-py-run":
        c_py_run(*sys.argv[2:])
    elif cmd == "child-cpp":
        c_cpp(*sys.argv[2:])
    elif cmd == "child-nontorch":
        c_nontorch(*sys.argv[2:])
    elif cmd == "python":
        missing = [m for m in ("torch", "matplotlib") if importlib.util.find_spec(m) is None]
        for dev, flags in DEVS:
            ids = [f"audio.{dev}::{name}" for name in CASES.values()] + [f"audio.{dev}::qa-run"]
            if missing:
                for tid in ids:
                    record(tid, "blocked", f"audio_unit_test.py needs {' and '.join(missing)} (not installed)")
                continue
            for case, name in CASES.items():
                py_child(f"audio.{dev}::{name}", "audio.py", ["child-py-case", str(case), *flags], timeout=660,
                         backend=dev)
            py_child(f"audio.{dev}::qa-run", "audio.py", ["child-py-run", *flags], timeout=1860, backend=dev)
    elif cmd == "cpp":
        binary = sys.argv[sys.argv.index("--bin") + 1]
        for dev, gpu in (("CPU", "0"), ("GPU", "1")):
            for case, name in CASES.items():
                if not os.access(binary, os.X_OK):
                    record(f"audio-cpp.{dev}::{name}", "blocked", "audio_tests did not build")
                    continue
                run_child(f"audio-cpp.{dev}::{name}", [os.environ.get("VP_PY", sys.executable), __file__,
                                                         "child-cpp", binary, str(case), gpu],
                          timeout=330, backend=dev)
    elif cmd == "nontorch":
        for dev in ("CPU", "GPU"):
            for case in ("audio_decoder", "preemphasis_filter"):
                py_child(f"audio-nontorch.{dev}::{case}", "audio.py", ["child-nontorch", case, dev], timeout=300,
                         backend=dev)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
