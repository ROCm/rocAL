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

"""rocAL readers matrix: one tiny pipeline per reader and backend (readers.<cpu|gpu>::<reader>).

    readers.py all                    every reader on both backends
    readers.py child <reader> <dev> [path]

Pass = the pipeline verifies, one batch comes out with the right shape and it is not
blank. Caffe/Caffe2 readers use the private LMDB copies in $VP_ROCAL_LMDB (M14);
their failure is C2. WebDataset (disabled in the build) and video (H2) are expected to
fail; TFRecord needs tensorflow (blocked otherwise).
"""
from __future__ import annotations

import importlib.util
import os
import sys

from common import data_root, fetch, lmdb_root, py_child, record, result

READERS = ("file", "coco", "caffe-lmdb", "caffe-lmdb-det", "caffe2-lmdb", "caffe2-lmdb-det", "mxnet", "numpy",
           "webdataset", "tfrecord", "video")
LMDB = {"caffe-lmdb": ("caffe", "classification"), "caffe-lmdb-det": ("caffe", "detection"),
        "caffe2-lmdb": ("caffe2", "classification"), "caffe2-lmdb-det": ("caffe2", "detection")}


def reader_images(kind: str, path: str | None = None):
    """Build the reader + decoder part inside an active pipeline; returns the image tensor."""
    import amd.rocal.fn as fn
    import amd.rocal.types as types
    d = data_root()
    dec = dict(device="cpu", max_decoded_width=416, max_decoded_height=416, output_type=types.RGB,
               random_shuffle=False)
    if kind == "file":
        root = str(d / "coco" / "coco_10_img" / "images")
        jpegs, _ = fn.readers.file(file_root=root)
        return fn.decoders.image(jpegs, file_root=root, **dec)
    if kind == "coco":
        root, ann = str(d / "coco" / "coco_10_img" / "images"), str(d / "coco" / "coco_10_img" / "annotations" /
                                                                    "coco_data.json")
        jpegs, _, _ = fn.readers.coco(annotations_file=ann)
        return fn.decoders.image(jpegs, file_root=root, annotations_file=ann, **dec)
    if kind in LMDB:
        fam, sub = LMDB[kind]
        p = path or str(lmdb_root() / fam / sub)
        reader = fn.readers.caffe if fam == "caffe" else fn.readers.caffe2
        outs = reader(path=p, bbox=sub == "detection")
        return fn.decoders.image(outs[0], path=p, **dec)
    if kind == "mxnet":
        p = str(d / "mxnet")
        return fn.decoders.image(fn.readers.mxnet(path=p), path=p, **dec)
    if kind == "numpy":
        return fn.readers.numpy(file_root=str(d / "numpy"), output_layout=types.NHWC, random_shuffle=False)
    if kind == "webdataset":
        p = str(d / "web_dataset" / "tar_file")
        return fn.decoders.image(fn.readers.webdataset(path=p, ext=[{"JPEG", "cls"}]), file_root=p, **dec)
    if kind == "tfrecord":
        import tensorflow as tf
        p = str(d / "tf" / "classification")
        keys = {"image/encoded": "image/encoded", "image/class/label": "image/class/label",
                "image/filename": "image/filename"}
        feats = {"image/encoded": tf.io.FixedLenFeature((), tf.string, ""),
                 "image/class/label": tf.io.FixedLenFeature([1], tf.int64, -1),
                 "image/filename": tf.io.FixedLenFeature((), tf.string, "")}
        inputs = fn.readers.tfrecord(p, keys, feats, reader_type=0)
        return fn.decoders.image(inputs["image/encoded"], user_feature_key_map=keys, path=p, output_type=types.RGB,
                                 max_decoded_width=416, max_decoded_height=416, random_shuffle=False)
    if kind == "video":
        vdir = str(d / "video_and_sequence_samples" / "labelled_videos" / "0")
        return fn.readers.video(file_root=vdir, sequence_length=3, random_shuffle=False)
    raise ValueError(kind)


def child(kind: str, dev: str, path: str | None = None) -> None:
    import amd.rocal.fn as fn
    import amd.rocal.types as types
    from probes import pipeline, safe_build
    pipe = pipeline(dev, tensor_layout=types.NHWC, tensor_dtype=types.UINT8)
    try:
        with pipe:
            images = reader_images(kind, path)
            pipe.set_outputs(images if kind == "video" else fn.resize(images, resize_width=64, resize_height=64))
    except RuntimeError as e:
        result("fail", f"reader setup failed: {e}")
        return
    if not safe_build(pipe):
        return
    try:
        out = fetch(pipe)
    except RuntimeError as e:
        result("fail", f"run failed: {e}")
        return
    blank = out.size == 0 or out.max() == out.min()
    result("fail" if blank else "pass", f"output {list(out.shape)} {out.dtype} mean={out.mean():.1f}"
           + (" (blank)" if blank else ""))


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(*sys.argv[2:])
        return 0
    have_tf = importlib.util.find_spec("tensorflow") is not None
    for dev in ("cpu", "gpu"):
        for kind in READERS:
            tid = f"readers.{dev}::{kind}"
            if kind == "tfrecord" and not have_tf:
                record(tid, "blocked", "tensorflow is not installed (fn.readers.tfrecord needs tf.io features)")
                continue
            py_child(tid, "readers.py", ["child", kind, dev], timeout=180, backend=dev.upper())
    return 0


if __name__ == "__main__":
    os.environ.setdefault("VP_ROCAL_LMDB", "")
    sys.exit(main())
