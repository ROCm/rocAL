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

"""amd.rocal API probes, each in its own child process (see common.run_child).

    probes.py smoke                 import + one tiny pipeline per backend (quick tier)
    probes.py all                   the API probes (comprehensive tier)
    probes.py rocjpeg --group G     the rocJPEG batch-6 control (run with and without the VA env)
    probes.py child <name> ...      internal: one probe body

Deliberate crash probes (M5 GPU, H9 batch > 6, the crash-instead-of-error items) run only
when VP_ROCAL_CRASH_OK=1, which run.sh sets from vp_crash_tests_allowed; otherwise they
are recorded as skip. The M12 probe runs under a short timeout because it hangs.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import threading
from pathlib import Path

from common import data_root, fetch, py_child, record, result, run_child, tiny_dataset

DEVS = ("cpu", "gpu")


# ---------------------------------------------------------------------------
# child helpers
# ---------------------------------------------------------------------------

def pipeline(dev: str, bs: int = 2, **kw):
    import amd.rocal.types as types
    from amd.rocal.pipeline import Pipeline
    opts = dict(batch_size=bs, num_threads=1, device_id=0, seed=1, rocal_cpu=(dev == "cpu"),
                tensor_layout=types.NHWC, tensor_dtype=types.UINT8)
    opts.update(kw)
    return Pipeline(**opts)


def safe_build(pipe) -> bool:
    """rocalVerify without Pipeline.build()'s exit(0) on failure (M8)."""
    import amd.rocal.types as types
    import rocal_pybind as b
    status = b.rocalVerify(pipe._handle)
    if status != types.OK:
        result("fail", f"rocalVerify failed (status {status}); see the [ERR] lines in the log")
        return False
    return True


def file_images(dev: str, root: Path, bs: int = 2, decoder: str = "cpu", **kw):
    import amd.rocal.fn as fn
    import amd.rocal.types as types
    pipe = pipeline(dev, bs, **kw)
    with pipe:
        jpegs, labels = fn.readers.file(file_root=str(root))
        images = fn.decoders.image(jpegs, file_root=str(root), output_type=types.RGB, device=decoder,
                                   random_shuffle=False)
    return pipe, images, labels


# ---------------------------------------------------------------------------
# probe bodies (children)
# ---------------------------------------------------------------------------

def c_smoke(dev: str) -> None:
    import amd.rocal.fn as fn
    pipe, images, _ = file_images(dev, tiny_dataset())
    with pipe:
        pipe.set_outputs(fn.resize(images, resize_width=64, resize_height=48))
    if not safe_build(pipe):
        return
    out = fetch(pipe)
    ok = list(out.shape) == [2, 48, 64, 3] and 20 < float(out.mean()) < 235
    result("pass" if ok else "fail", f"shape={list(out.shape)} mean={out.mean():.1f}")


def c_crop_pos(dev: str) -> None:
    """M7: fn.crop(crop_pos_x=0.5, crop_pos_y=0.5) must crop the centre."""
    import amd.rocal.fn as fn
    import numpy as np
    from PIL import Image
    root = tiny_dataset()
    pipe, images, _ = file_images(dev, root)
    with pipe:
        pipe.set_outputs(fn.crop(images, crop=(200, 200), crop_pos_x=0.5, crop_pos_y=0.5))
    if not safe_build(pipe):
        return
    out = fetch(pipe)[0].astype(np.int16)
    src = np.asarray(Image.open(root / sorted(os.listdir(root))[0]).convert("RGB")).astype(np.int16)
    h, w = src.shape[:2]
    ex, ey = round(0.5 * (w - 200)), round(0.5 * (h - 200))
    mad_exp = float(np.abs(src[ey:ey + 200, ex:ex + 200] - out).mean())
    mad_00 = float(np.abs(src[:200, :200] - out).mean())
    msg = f"input {w}x{h}: expected offset ({ex},{ey}) MAD={mad_exp:.2f}; offset (0,0) MAD={mad_00:.2f}"
    if mad_exp < 3:
        result("pass", msg)
    elif mad_00 < 3:
        result("fail", "crop_pos_x/crop_pos_y ignored, crop taken at (0,0) (M7); " + msg)
    else:
        result("fail", "crop matches neither the expected nor the (0,0) offset; " + msg)


def c_log(op: str, dtype: str, dev: str) -> None:
    """H8: fn.log / fn.log1p against numpy on numpy-reader inputs."""
    import amd.rocal.fn as fn
    import amd.rocal.types as types
    import numpy as np
    d = Path.cwd() / f"in_{dtype}"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(2):
        np.save(d / f"{i}.npy", rng.uniform(0.5, 200.0, size=(32, 32, 3)).astype(dtype))
    pipe = pipeline(dev, tensor_dtype=types.FLOAT)
    ops = {"log": (fn.log, np.log), "log1p": (fn.log1p, np.log1p), "copy": (fn.copy, lambda a: a)}
    with pipe:
        x = fn.readers.numpy(file_root=str(d), output_layout=types.NHWC, random_shuffle=False)
        pipe.set_outputs(ops[op][0](x))
    if not safe_build(pipe):
        return
    out = fetch(pipe).astype(np.float64)
    src = np.stack([np.load(d / f"{i}.npy") for i in range(2)]).astype(np.float64)
    ref = ops[op][1](src)
    good = np.isclose(out.reshape(ref.shape), ref, atol=1e-3)
    msg = (f"{op}({dtype}) on {dev}: {good.mean() * 100:.1f}% within 1e-3 of numpy, "
           f"nan={int(np.isnan(out).sum())}, nonzero={float((out != 0).mean() * 100):.1f}%")
    result("pass" if good.all() else "fail", msg + ("" if good.all() else " (H8)"))


def c_one_hot(dev: str) -> None:
    """M11: one-hot labels, CPU and GPU."""
    import amd.rocal.fn as fn
    import numpy as np
    from amd.rocal.plugin.generic import ROCALClassificationIterator
    root = data_root() / "images_jpg" / "labels_folder"
    classes = len(next(os.walk(root))[1])
    pipe, images, labels = file_images(dev, root)
    with pipe:
        pipe.set_outputs(fn.crop(images, crop=(224, 224)))
        fn.one_hot(labels, num_classes=classes)
    if not safe_build(pipe):
        return
    it = ROCALClassificationIterator(pipe, device="cpu" if dev == "cpu" else "gpu", device_id=0)
    try:
        _, lab = next(iter(it))
    except RuntimeError as e:
        result("fail", f"one-hot labels failed: {e} (M11)")
        return
    lab = np.asarray(lab).reshape(-1, classes)
    ok = lab.shape == (2, classes) and np.all(lab.sum(axis=1) == 1)
    result("pass" if ok else "fail", f"labels shape {lab.shape}: {lab.tolist()}")


def c_symlink(mode: str) -> None:
    """M10: a labelled dataset whose files are symlinks (mode=symlink) or copies (control)."""
    from amd.rocal.plugin.generic import ROCALClassificationIterator
    root = Path.cwd() / f"ds_{mode}"
    src = tiny_dataset()
    for label in ("0", "1"):
        (root / label).mkdir(parents=True, exist_ok=True)
        for name in sorted(os.listdir(src))[int(label) * 4:int(label) * 4 + 4]:
            dst = root / label / name
            if dst.exists():
                continue
            if mode == "symlink":
                dst.symlink_to(src / name)
            else:
                shutil.copy(src / name, dst)
    pipe, images, _ = file_images("cpu", root)
    with pipe:
        pipe.set_outputs(images)
    if not safe_build(pipe):
        return
    n = 0
    try:
        for _out, _lab in ROCALClassificationIterator(pipe, device="cpu"):
            n += 1
    except RuntimeError as e:
        result("fail", f"label reader failed on a {mode} dataset: {e}" + (" (M10)" if mode == "symlink" else ""))
        return
    result("pass" if n == 4 else "fail", f"{n} batches of 2 from 8 {mode}ed files")


def c_extsrc(mode: str, dev: str) -> None:
    """M12: external_source_reader.py builds its data dir as ROCAL_DATA_PATH + 'rocal_data/...'."""
    import glob

    import amd.rocal.fn as fn
    import amd.rocal.types as types
    import numpy as np
    from amd.rocal.plugin.generic import ROCALClassificationIterator
    base = os.environ["VP_DATA"].rstrip("/") + ("/" if mode == "sep" else "")
    data_dir = base + "rocal_data/coco/coco_10_img/images/"

    class Source:  # the upstream ExternalInputIteratorMode0, minus the shuffle
        def __init__(self, bs):
            self.bs = bs
            self.files = sorted(f for p in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
                                for f in glob.glob(os.path.join(data_dir, p)))

        def __iter__(self):
            self.i, self.n = 0, len(self.files)
            return self

        def __next__(self):
            batch, labels = [], []
            for k in range(self.bs):
                batch.append(self.files[self.i])
                labels.append(k + 1)
                self.i = (self.i + 1) % self.n
            return batch, np.array(labels).astype("int32")

    src = Source(2)
    print(f"{len(src.files)} files under {data_dir}", flush=True)
    pipe = pipeline(dev, prefetch_queue_depth=4, tensor_layout=types.NCHW)
    with pipe:
        jpegs, _ = fn.external_source(source=src, mode=types.EXTSOURCE_FNAME)
        pipe.set_outputs(fn.resize(jpegs, resize_width=300, resize_height=300, output_layout=types.NCHW,
                                   output_dtype=types.UINT8))
    if not safe_build(pipe):
        return
    try:
        n = 0
        for _ in ROCALClassificationIterator(pipe, device=dev):
            n += 1
            if n >= 3:
                break
    except (RuntimeError, IndexError, ZeroDivisionError) as e:
        result("pass" if mode == "nosep" else "fail", f"{len(src.files)} files; raised {type(e).__name__}: {e}")
        return
    result("pass" if n else "fail", f"{len(src.files)} files; {n} batches")


BUILD_FAIL_CANDIDATES = ("color-jitter-gpu", "preemphasis-cpu")


def c_build_exit() -> None:
    """M8: Pipeline.build() must not exit(0) when rocalVerify fails.

    Needs a graph that fails verification; the candidates are known-failing graphs
    (color_jitter on GPU, audio pre-emphasis under H5). If every candidate verifies,
    the probe cannot run and is skipped.
    """
    import subprocess
    for cand in BUILD_FAIL_CANDIDATES:
        p = subprocess.run([sys.executable, __file__, "child", "build-exit-inner", cand], capture_output=True,
                           text=True, timeout=120)
        out = p.stdout + p.stderr
        if "INNER-AFTER-BUILD" in out:
            continue
        if "Verify graph failed" in out and p.returncode == 0:
            result("fail", f"[{cand}] Pipeline.build() printed 'Verify graph failed' and exited with status 0 (M8)")
        elif p.returncode != 0:
            result("pass", f"[{cand}] failed verify surfaced as exit {p.returncode}")
        else:
            result("fail", f"[{cand}] exit 0 without reaching the code after build(); {out[-300:]}")
        return
    result("skip", f"could not provoke a verify failure: {', '.join(BUILD_FAIL_CANDIDATES)} all verified")


def c_build_exit_inner(cand: str) -> None:
    import amd.rocal.fn as fn
    from amd.rocal.pipeline import Pipeline
    if cand == "color-jitter-gpu":
        pipe, images, _ = file_images("gpu", tiny_dataset())
        with pipe:
            pipe.set_outputs(fn.color_jitter(images))
    else:
        audio = data_root() / "audio"
        pipe = Pipeline(batch_size=1, num_threads=1, device_id=0, seed=0, rocal_cpu=True)
        with pipe:
            wav, _ = fn.readers.file(file_root=str(audio) + "/", file_list=str(audio / "wav_file_list.txt"))
            dec = fn.decoders.audio(wav, file_root=str(audio) + "/", file_list_path=str(audio / "wav_file_list.txt"),
                                    downmix=False, shard_id=0, num_shards=1, stick_to_shard=False)
            pipe.set_outputs(fn.preemphasis_filter(dec))
    pipe.build()
    print("INNER-AFTER-BUILD", flush=True)


def _concurrent_pipe(dev: str):
    import amd.rocal.fn as fn
    import amd.rocal.types as types
    pipe, images, _ = file_images(dev, tiny_dataset(), bs=8, num_threads=2, seed=3)
    with pipe:
        img = fn.resize(images, resize_width=160, resize_height=160)
        img = fn.flip(img, horizontal=1)
        pipe.set_outputs(fn.brightness(img, brightness=1.2, brightness_shift=0.0))
    if not safe_build(pipe):
        raise RuntimeError("verify failed")
    return pipe, types


def _digest(pipe, dev: str):
    import numpy as np
    from amd.rocal.plugin.generic import ROCALClassificationIterator
    h, n = hashlib.sha256(), 0
    for out, lab in ROCALClassificationIterator(pipe, device=dev):
        a = np.ascontiguousarray(np.array(out[0]))
        h.update(a.tobytes())
        h.update(np.array(lab, dtype=np.int64).tobytes())
        n += a.shape[0]
    return h.hexdigest(), n


def c_concurrency(mode: str, dev: str) -> None:
    """M5: 4 pipelines built (mode=build) or only iterated (mode=prebuilt) in threads vs a solo run."""
    ref = _digest(_concurrent_pipe(dev)[0], dev)
    res: list = [None] * 4
    pipes = [_concurrent_pipe(dev)[0] for _ in range(4)] if mode == "prebuilt" else [None] * 4

    def worker(i):
        try:
            res[i] = _digest(pipes[i] if pipes[i] is not None else _concurrent_pipe(dev)[0], dev)
        except BaseException as e:  # noqa: BLE001 - any failure in a thread is a probe failure
            res[i] = (f"ERR {e!r}", 0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    same = sum(1 for r in res if r == ref)
    msg = f"{same}/4 concurrent runs identical to the solo run ({ref[1]} images each)"
    result("pass" if same == 4 else "fail", msg + ("" if same == 4 else f" (M5); results={[r[0][:12] for r in res]}"))


def c_rocjpeg(bs: str) -> None:
    """H9: rocJPEG (hardware) decode with `bs` images in flight."""
    import amd.rocal.fn as fn
    import numpy as np
    from amd.rocal.plugin.generic import ROCALClassificationIterator
    pipe, images, _ = file_images("gpu", tiny_dataset(), bs=int(bs), decoder="gpu")
    with pipe:
        pipe.set_outputs(fn.resize(images, resize_width=224, resize_height=224))
    if not safe_build(pipe):
        return
    n, s = 0, 0.0
    for out, _ in ROCALClassificationIterator(pipe, device="gpu"):
        s += float(np.array(out[0]).mean())
        n += 1
    ok = n == -(-256 // int(bs)) and 20 < s / max(n, 1) < 235
    result("pass" if ok else "fail", f"batch {bs}: {n} batches, mean {s / max(n, 1):.2f}")


def c_cpu_threads() -> None:
    """H7: a CPU resize pipeline must give identical batches with 1 and 4 CPU threads."""
    import amd.rocal.fn as fn
    outs = {}
    for threads in (1, 4):
        pipe, images, _ = file_images("cpu", tiny_dataset(), bs=4, num_threads=threads)
        with pipe:
            pipe.set_outputs(fn.resize(images, resize_width=224, resize_height=224))
        if not safe_build(pipe):
            return
        outs[threads] = fetch(pipe)
    a, b = outs[1], outs[4]
    diff = [int((a[i] != b[i]).sum()) for i in range(a.shape[0])]
    zero_heads = [i for i in range(b.shape[0]) if not b[i].reshape(-1)[:8].any() and a[i].reshape(-1)[:8].any()]
    msg = f"bytes differing per image (1 vs 4 threads): {diff}; images whose first 8 bytes became 0: {zero_heads}"
    result("pass" if not any(diff) else "fail", msg + ("" if not any(diff) else " (H7)"))


def c_color_jitter(dev: str) -> None:
    """Low item: color_jitter has no GPU path."""
    import amd.rocal.fn as fn
    pipe, images, _ = file_images(dev, tiny_dataset())
    with pipe:
        pipe.set_outputs(fn.color_jitter(images))
    if not safe_build(pipe):
        return
    out = fetch(pipe)
    result("pass" if out.max() > out.min() else "fail", f"output mean {out.mean():.1f}")


def c_crash_log1p_float32(dev: str) -> None:
    """Low item: log1p on float32 should raise a clean error; it segfaults."""
    import amd.rocal.fn as fn
    import amd.rocal.types as types
    import numpy as np
    d = Path.cwd() / "in"
    d.mkdir(exist_ok=True)
    for i in range(2):
        np.save(d / f"{i}.npy", np.full((8, 8, 3), 2.0, dtype=np.float32))
    pipe = pipeline(dev, tensor_dtype=types.FLOAT)
    try:
        with pipe:
            x = fn.readers.numpy(file_root=str(d), output_layout=types.NHWC)
            pipe.set_outputs(fn.log1p(x))
        pipe.build()
    except RuntimeError as e:
        result("pass", f"clean error: {e}")
        return
    result("fail", "log1p on float32 was accepted")


def c_crash_cpu_gpu_decoder() -> None:
    """Low item: a CPU pipeline with the GPU (rocJPEG) decoder should raise, not abort."""
    try:
        pipe, images, _ = file_images("cpu", tiny_dataset(), decoder="gpu")
        with pipe:
            pipe.set_outputs(images)
        if not safe_build(pipe):
            return
        fetch(pipe)
    except RuntimeError as e:
        result("pass", f"clean error: {e}")
        return
    result("pass", "ran")


def c_box_encoder(binary: str, gpu: str, rgb: str) -> None:
    """M6: the SSD box-encoder pipeline of testAllScripts.sh (reader 26) crashes in rocalRelease."""
    import subprocess
    images = str(data_root() / "coco" / "coco_10_img" / "images") + "/"
    p = subprocess.run([binary, "26", images, str(Path.cwd() / "CropResizeRandom"), "416", "416", "1", gpu, rgb,
                        "1", "0"], capture_output=True, text=True, timeout=240)
    print((p.stdout + p.stderr)[-5000:], flush=True)
    if p.returncode == 0:
        result("pass", "box-encoder pipeline ran and released cleanly")
    elif p.returncode < 0:
        result("error", f"killed by signal {-p.returncode} (M6: segfault in MasterGraph::release)")
    else:
        result("fail", f"exit {p.returncode}")


def c_extended(kind: str) -> None:
    """Extended image only: the torch / jax / tensorflow plugins on one GPU batch."""
    import amd.rocal.fn as fn
    import numpy as np
    if kind == "tensorflow":
        import tensorflow  # noqa: F401
        from amd.rocal.plugin.tf import ROCALIterator as It
    elif kind == "jax":
        from amd.rocal.plugin.jax import ROCALJaxIterator as It
    else:
        import torch
        from amd.rocal.plugin.pytorch import ROCALClassificationIterator as It
        if not torch.cuda.is_available():
            result("fail", "torch.cuda.is_available() is False (not a ROCm torch?)")
            return
    pipe, images, _ = file_images("gpu", tiny_dataset())
    with pipe:
        pipe.set_outputs(fn.resize(images, resize_width=64, resize_height=64))
    if not safe_build(pipe):
        return
    batch = next(iter(It(pipe, device="gpu") if kind != "jax" else It(pipe)))
    arr = batch[0][0] if isinstance(batch[0], (list, tuple)) else batch[0]
    result("pass", f"{kind} plugin batch: {type(arr).__name__} {tuple(np.shape(arr))}")


CHILDREN = {
    "smoke": c_smoke, "crop-pos": c_crop_pos, "log": c_log, "one-hot": c_one_hot, "symlink": c_symlink,
    "extsrc": c_extsrc, "build-exit": c_build_exit, "build-exit-inner": c_build_exit_inner,
    "concurrency": c_concurrency, "rocjpeg": c_rocjpeg, "color-jitter": c_color_jitter,
    "crash-log1p-float32": c_crash_log1p_float32, "crash-cpu-gpu-decoder": c_crash_cpu_gpu_decoder,
    "box-encoder": c_box_encoder, "extended": c_extended, "cpu-threads": c_cpu_threads,
}


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------

def crash_ok() -> bool:
    return os.environ.get("VP_ROCAL_CRASH_OK") == "1"


def crash_probe(test_id: str, args: list[str], **kw) -> None:
    if crash_ok():
        py_child(test_id, "probes.py", ["child", *args], **kw)
    else:
        record(test_id, "skip", "deliberate crash probe: skipped on a bare host (vp_crash_tests_allowed)")


def be(dev: str) -> str:
    return "GPU" if dev == "gpu" else "CPU"


def smoke() -> None:
    for mod in ("rocal_pybind", "amd.rocal.pipeline", "amd.rocal.fn", "amd.rocal.plugin.generic"):
        py_child(f"smoke::import.{mod}", "probes.py", ["child-import", mod], timeout=120)
    for dev in DEVS:
        py_child(f"smoke::pipeline.{dev}", "probes.py", ["child", "smoke", dev], timeout=180, backend=be(dev))


def all_probes() -> None:
    for dev in DEVS:
        py_child(f"probe::crop-pos.{dev}", "probes.py", ["child", "crop-pos", dev], backend=be(dev))
    for op, dt in (("copy", "float32"), ("copy", "int16"), ("log", "float32"), ("log", "int16"), ("log1p", "int16")):
        for dev in DEVS:
            name = f"log-control-copy.{dt}" if op == "copy" else f"{op}.{dt}"
            py_child(f"probe::{name}.{dev}", "probes.py", ["child", "log", op, dt, dev], backend=be(dev))
    for dev in DEVS:
        py_child(f"probe::one-hot.{dev}", "probes.py", ["child", "one-hot", dev], backend=be(dev))
    # M10: rocAL skips the symlinks, finds no files and then segfaults while building the graph.
    crash_probe("probe::labels-symlink-dataset.cpu", ["symlink", "symlink"], backend="CPU")
    py_child("probe::labels-copy-dataset.cpu", "probes.py", ["child", "symlink", "copy"], backend="CPU")
    for mode in ("nosep", "sep"):
        py_child(f"probe::external-source-{mode}.cpu", "probes.py", ["child", "extsrc", mode, "cpu"],
                 timeout=60, backend="CPU", expect_hang=(mode == "nosep"))
    py_child("probe::build-exit-status", "probes.py", ["child", "build-exit"], timeout=180, backend="CPU")
    py_child("probe::concurrent-build.cpu", "probes.py", ["child", "concurrency", "build", "cpu"],
             timeout=300, backend="CPU")
    crash_probe("probe::concurrent-build.gpu", ["concurrency", "build", "gpu"], timeout=300, backend="GPU")
    for dev in DEVS:
        py_child(f"probe::concurrent-prebuilt.{dev}", "probes.py", ["child", "concurrency", "prebuilt", dev],
                 timeout=300, backend=be(dev))
    py_child("probe::cpu-threads-resize.cpu", "probes.py", ["child", "cpu-threads"], backend="CPU")
    crash_probe("probe::rocjpeg-batch8.gpu", ["rocjpeg", "8"], timeout=180, backend="GPU")
    for dev in DEVS:
        py_child(f"probe::color-jitter.{dev}", "probes.py", ["child", "color-jitter", dev], backend=be(dev))
    for dev in DEVS:
        crash_probe(f"probe::crash.log1p-float32.{dev}", ["crash-log1p-float32", dev], timeout=120, backend=be(dev))
    crash_probe("probe::crash.cpu-pipeline-gpu-decoder", ["crash-cpu-gpu-decoder"], timeout=120, backend="CPU")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "child":
        CHILDREN[sys.argv[2]](*sys.argv[3:])
        return 0
    if cmd == "child-import":
        import importlib
        m = importlib.import_module(sys.argv[2])
        result("pass", f"imported {sys.argv[2]} from {getattr(m, '__file__', '?')}")
        return 0
    if cmd == "smoke":
        smoke()
    elif cmd == "all":
        all_probes()
    elif cmd == "rocjpeg":
        group = sys.argv[sys.argv.index("--group") + 1] if "--group" in sys.argv else "probe"
        py_child(f"{group}::rocjpeg-batch6.gpu", "probes.py", ["child", "rocjpeg", "6"], timeout=180, backend="GPU")
    elif cmd == "cpp":
        binary = sys.argv[sys.argv.index("--bin") + 1]
        for what in ("nop", "copy-size"):
            for dev in DEVS:
                name = "nop-uninitialized" if what == "nop" else "copy-to-output-size"
                if not os.access(binary, os.X_OK):
                    record(f"probe::{name}.{dev}", "blocked", "api_probe did not build")
                    continue
                run_child(f"probe::{name}.{dev}", [binary, what, dev], timeout=180, backend=be(dev))
    elif cmd == "box-encoder":
        binary = sys.argv[sys.argv.index("--bin") + 1]
        for dev, gpu in (("cpu", "0"), ("gpu", "1")):
            for color, rgb in (("gray", "0"), ("rgb", "1")):
                crash_probe(f"probe::box-encoder-release.{dev}.{color}", ["box-encoder", binary, gpu, rgb],
                            timeout=300, backend=be(dev))
    elif cmd == "extended":
        for kind in ("torch", "jax", "tensorflow"):
            tid = f"extended::{kind}-plugin.gpu"
            if os.environ.get("VP_EXTENDED") != "1":
                record(tid, "blocked", "needs the extended image (ROCm torch, jax, tensorflow): VP_EXTENDED=1")
            else:
                py_child(tid, "probes.py", ["child", "extended", kind], timeout=600, backend="GPU")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
