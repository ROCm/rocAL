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

"""amd.rocal file-reader -> decode [-> resize] pipeline against a PIL reference; judge the outcome (verdict.py).

    rocal_pipeline.py --backend cpu|gpu --mode correct|honest|error --data DIR [--resize W H]

Batch 4, one thread (multi-threaded CPU resize corrupts image heads, H7), no shuffle, NHWC uint8.
The reference is PIL decode (+ bilinear resize) of the first files in sorted order; decode-only output
is compared on each image's own region. Pipeline.build() calls exit(0) when rocalVerify fails (M8), so
a SystemExit(0) from build() is reported as "exit0_error".
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from verdict import finish

BATCH = 4
MAX_MAD = 4.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["cpu", "gpu"])
    ap.add_argument("--mode", required=True, choices=["correct", "honest", "error"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"))
    a = ap.parse_args()
    data = a.data.rstrip("/") + "/"
    try:
        import amd.rocal.fn as fn
        import amd.rocal.types as types
        from amd.rocal.pipeline import Pipeline
        from amd.rocal.plugin.generic import ROCALClassificationIterator
    except Exception as e:  # noqa: BLE001
        finish(a.mode, "clean_error", f"import amd.rocal: {type(e).__name__}: {e}")

    try:
        pipe = Pipeline(batch_size=BATCH, num_threads=1, device_id=0, seed=1, rocal_cpu=a.backend == "cpu",
                        tensor_layout=types.NHWC, tensor_dtype=types.UINT8)
        with pipe:
            jpegs, _ = fn.readers.file(file_root=data)
            images = fn.decoders.image(jpegs, file_root=data, output_type=types.RGB, random_shuffle=False)
            if a.resize:
                images = fn.resize(images, resize_width=a.resize[0], resize_height=a.resize[1])
            pipe.set_outputs(images)
        try:
            pipe.build()
        except SystemExit as e:
            if e.code in (0, None):
                finish(a.mode, "exit0_error", "Pipeline.build() failed verification and called exit(0) (M8)")
            finish(a.mode, "clean_error", f"Pipeline.build() exited {e.code}")
        print("build OK", flush=True)
        it = ROCALClassificationIterator(pipe, device=a.backend)
        batch, _ = next(iter(it))
        got = np.array(batch[0])
    except Exception as e:  # noqa: BLE001 - the library reported an error
        finish(a.mode, "clean_error", f"{type(e).__name__}: {e}")
    print(f"output shape {got.shape} dtype {got.dtype} mean {got.mean():.2f}", flush=True)

    from PIL import Image
    names = sorted(f for f in os.listdir(data) if f.lower().endswith((".jpg", ".jpeg")))[:BATCH]
    mads = []
    for i, name in enumerate(names):
        im = Image.open(data + name).convert("RGB")
        if a.resize:
            im = im.resize(tuple(a.resize), Image.BILINEAR)
        ref = np.asarray(im).astype(np.int16)
        h, w = ref.shape[:2]
        if got.ndim != 4 or i >= got.shape[0] or got.shape[1] < h or got.shape[2] < w:
            finish(a.mode, "wrong", f"output shape {got.shape} cannot hold reference image {name} {ref.shape}")
        mads.append(float(np.abs(got[i, :h, :w, :].astype(np.int16) - ref).mean()))
    print("mean abs diff vs PIL per image:", [round(m, 2) for m in mads], flush=True)
    bad = [m for m in mads if m > MAX_MAD]
    finish(a.mode, "wrong" if bad else "correct",
           f"{len(bad)}/{len(mads)} images above MAD {MAX_MAD} vs PIL (max {max(mads):.2f})")


if __name__ == "__main__":
    main()
