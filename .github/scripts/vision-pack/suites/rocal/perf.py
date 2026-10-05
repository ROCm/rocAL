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

"""rocAL throughput (full tier): performance_tests and dataloader_multithread.

    perf.py --pt BIN --dm BIN --data DIR --out perf.json

Runs sequentially on a dataset of real JPEG copies (symlinked datasets break the label
readers, M10). Records rocal::perf::<metric> (pass when the run finished and printed its
timing) and writes {"metrics": [...]} for vp_perf. rocJPEG runs stay at batch 4 with one
thread: more than 6 images in flight fails or crashes on gfx1201 (H9).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from common import record, rel, slug

PT_CASES = (0, 1, 2, 3, 7)
PT_BATCHES = (16, 64)
# name: (num_gpus, batch, dec_mode, cpu_threads); dec_mode 0 = TurboJPEG, 4 = rocJPEG
DM_CONFIGS = {"cpu-tjpeg-bs16-t8": (0, 16, 0, 8), "gpu-tjpeg-bs16-t8": (1, 16, 0, 8),
              "gpu-rocjpeg-bs4-t1": (1, 4, 4, 1)}


def run(cmd: list[str], log: Path, timeout: int = 600) -> tuple[int, str, float]:
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc, text = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        rc, text = 124, "timeout"
    log.write_text(f"### cmd: {' '.join(cmd)}\n{text}")
    return rc, text, time.monotonic() - t0


def elapsed_us(text: str) -> int | None:
    m = re.findall(r"Total Elapsed Time:? (\d+) sec (\d+) us", text)
    return int(m[-1][0]) * 1_000_000 + int(m[-1][1]) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--dm", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n_images = sum(1 for _ in Path(a.data).rglob("*.JPEG"))
    logs = Path(os.environ["VP_OUT"]) / "logs"
    metrics = []

    def emit(name, backend, rc, text, dur, images, log):
        us = elapsed_us(text)
        tid = f"perf::{name}.{backend}"
        if rc != 0 or not us:
            status = "error" if rc in (124, 137) or rc < 0 else "fail"
            record(tid, status, f"exit {rc}, timing {'found' if us else 'missing'}", duration_s=dur, log=rel(str(log)),
                   backend=backend)
            return
        ips = images / (us / 1e6)
        metrics.append({"name": f"rocal.{name}", "value": round(ips, 1), "unit": "images/s",
                        "lower_is_better": False, "backend": backend})
        record(tid, "pass", f"{ips:.0f} images/s ({images} images in {us / 1e6:.3f} s)", duration_s=dur,
               log=rel(str(log)), backend=backend)

    for tc in PT_CASES:
        for bs in PT_BATCHES:
            for gpu, backend in (("0", "CPU"), ("1", "GPU")):
                log = logs / f"{slug(f'perf-pt-{tc}-{bs}-{backend}')}.log"
                rc, text, dur = run([a.pt, a.data, "224", "224", str(tc), str(bs), gpu, "1", "1", "0"], log)
                m = re.search(r"Running\s+(rocal\w+)", text)
                aug = m.group(1) if m else f"case{tc}"
                # performance_tests stops after 100 batches.
                emit(f"performance_tests.{aug}.bs{bs}", backend, rc, text, dur, min(100 * bs, n_images), log)
    for name, (gpus, bs, dec, thr) in DM_CONFIGS.items():
        log = logs / f"{slug(f'perf-dm-{name}')}.log"
        rc, text, dur = run([a.dm, a.data, str(gpus), "1", "224", "224", str(bs), "0", "0", str(dec), str(thr)], log)
        m = re.findall(r"Processed (\d+) images", text)
        emit(f"dataloader_multithread.{name}", "GPU" if gpus else "CPU", rc, text, dur,
             int(m[-1]) if m else n_images, log)
    Path(a.out).write_text(json.dumps({"metrics": metrics}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
