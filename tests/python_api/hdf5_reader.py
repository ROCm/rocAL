# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""HDF5 input, numerical, failure and lifecycle checks using generated fixtures."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

import amd.rocal.fn as fn
import amd.rocal.types as types
from amd.rocal.pipeline import Pipeline


def create_fixture(root, count=8, dtype="float32"):
    for i in range(count):
        with h5py.File(root / f"sample-{i:02d}.h5", "w") as f:
            f["data"] = (np.arange(24).reshape(2, 3, 4) + 100 * i).astype(dtype)
            f["label"] = np.full((2, 3), i, dtype=np.int32)


def make_pipeline(root, cpu, batch=2, keys=None, **options):
    pipe = Pipeline(
        batch_size=batch,
        num_threads=1,
        rocal_cpu=cpu,
        output_memory_type=types.HOST_MEMORY,
        prefetch_queue_depth=2,
    )
    try:
        with pipe:
            outputs = fn.readers.hdf5(
                file_root=str(root), dataset_keys=keys or ["data", "label"], **options
            )
            pipe.set_outputs(*outputs)
    except Exception:
        pipe.rocal_release()
        raise
    return pipe


def read_batch(pipe, partial=False):
    if pipe.rocal_run() != 0:
        return None
    outputs = []
    for tensor in pipe.get_output_tensors():
        data = np.empty(tensor.dimensions(), dtype=tensor.dtype())
        tensor.copy_data(data)
        if partial and pipe.get_remaining_images() < pipe._batch_size:
            valid = pipe._batch_size - pipe.get_last_batch_padded_size()
            data = data[:valid]
        outputs.append(data)
    return outputs


def consume(pipe, root, keys=("data", "label"), partial=False):
    ids = []
    while True:
        output = read_batch(pipe, partial)
        if output is None:
            return ids
        for row in range(len(output[0])):
            # Each fixture stores its numeric file ID in every label
            # coordinate.
            sample = (
                int(output[keys.index("label")][row].flat[0])
                if "label" in keys
                else int(output[0][row].flat[0] // 100)
            )
            with h5py.File(root / f"sample-{sample:02d}.h5", "r") as f:
                for key, data in zip(keys, output):
                    np.testing.assert_array_equal(data[row], f[key][:])
                    assert data.dtype == f[key].dtype.newbyteorder("=")
            ids.append(sample)


def run_reader(root, cpu, keys=("data", "label"), **options):
    pipe = make_pipeline(root, cpu, keys=list(keys), **options)
    try:
        pipe.build()
        return consume(
            pipe,
            root,
            keys,
            options.get("last_batch_policy") == types.LAST_BATCH_PARTIAL,
        )
    finally:
        pipe.rocal_release()


def rejected(action, text):
    try:
        action()
    except (RuntimeError, ValueError) as error:
        assert text.lower() in str(error).lower(), str(error)
    else:
        raise AssertionError(f"Expected rejection: {text}")


def values(root, cpu):
    create_fixture(root)
    assert run_reader(root, cpu) == list(range(8))
    assert run_reader(root, cpu, keys=("label", "data")) == list(range(8))
    assert run_reader(root, cpu, keys=("data",)) == list(range(8))
    assert run_reader(root, cpu, output_layouts=[types.NHWC, types.NONE]) == list(
        range(8)
    )
    # Two native workers run concurrently, including while another schema is
    # inspected.
    pipes = [make_pipeline(root, cpu) for _ in range(2)]
    try:
        for p in pipes:
            p.build()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda p: consume(p, root), pipes))
        assert results == [list(range(8))] * 2
    finally:
        for p in pipes:
            p.rocal_release()


def layouts(root, cpu):
    create_fixture(root, count=2)
    cases = [(types.NONE, (2,) * rank) for rank in range(1, 5)]
    cases += [(layout, (2, 3)) for layout in (types.NHW, types.NFT, types.NTF)]
    cases += [(layout, (2, 3, 4)) for layout in (types.NHWC, types.NCHW)]
    cases += [(layout, (2, 3, 4, 2)) for layout in (types.NDHWC, types.NCDHW)]
    for layout, shape in cases:
        for i in range(2):
            with h5py.File(root / f"sample-{i:02d}.h5", "a") as f:
                del f["data"]
                f["data"] = (np.arange(np.prod(shape)).reshape(shape) + 100 * i).astype(
                    "float32"
                )
        assert run_reader(root, cpu, output_layouts=[layout, types.NONE]) == [0, 1]


def file_order(root, cpu):
    create_fixture(root)
    chosen = [5, 0, 7, 1]
    files = [f"sample-{i:02}.h5" for i in chosen]
    assert run_reader(root, cpu, files=files) == chosen
    rejected(
        lambda: make_pipeline(root, cpu, files=[files[0], "./" + files[0]]), "duplicate"
    )


def batches(root, cpu):
    create_fixture(root, count=5)
    assert run_reader(root, cpu, last_batch_policy=types.LAST_BATCH_DROP) == [
        0,
        1,
        2,
        3,
    ]
    assert run_reader(root, cpu, last_batch_policy=types.LAST_BATCH_PARTIAL) == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert run_reader(root, cpu, pad_last_batch=True) == [0, 1, 2, 3, 4, 4]
    assert run_reader(root, cpu, pad_last_batch=False) == [0, 1, 2, 3, 4, 0]
    assert run_reader(root, cpu, batch=8, last_batch_policy=types.LAST_BATCH_DROP) == []
    assert run_reader(
        root, cpu, batch=8, last_batch_policy=types.LAST_BATCH_PARTIAL
    ) == list(range(5))


def shards(root, cpu):
    create_fixture(root, count=10)
    results = [run_reader(root, cpu, num_shards=3, shard_id=i) for i in range(3)]
    assert results == [[0, 3, 6, 9], [1, 4, 7, 1], [2, 5, 8, 2]], results
    assert len(set(map(len, results))) == 1
    assert set(sum(results, [])) == set(range(10))
    for i in range(3):
        rejected(
            lambda: make_pipeline(
                root,
                cpu,
                num_shards=3,
                shard_id=i,
                last_batch_policy=types.LAST_BATCH_DROP,
            ),
            "unequal batch counts",
        )
    assert [
        run_reader(
            root,
            cpu,
            num_shards=3,
            shard_id=i,
            shard_size=2,
            last_batch_policy=types.LAST_BATCH_DROP,
        )
        for i in range(3)
    ] == [[0, 3], [1, 4], [2, 5]]
    for opts in (
        {"num_shards": 0},
        {"num_shards": 2, "shard_id": 2},
        {"num_shards": 11},
        {"shard_size": 0},
    ):
        rejected(lambda: make_pipeline(root, cpu, **opts), "shard")


def epochs(root, cpu):
    create_fixture(root, count=12)

    def sequence(**options):
        pipe = make_pipeline(root, cpu, num_shards=2, shard_id=0, **options)
        try:
            pipe.build()
            results = []
            for epoch in range(3):
                results.append(consume(pipe, root))
                assert pipe.rocal_reset_loaders() == 0
            return results
        finally:
            pipe.rocal_release()

    fixed = sequence(stick_to_shard=True)
    assert fixed == [list(range(0, 12, 2))] * 3
    rotating = sequence(stick_to_shard=False)
    assert rotating == [
        list(range(0, 12, 2)),
        list(range(1, 12, 2)),
        list(range(0, 12, 2)),
    ]
    shuffled = sequence(random_shuffle=True, seed=7)
    assert shuffled == sequence(random_shuffle=True, seed=7)
    assert shuffled[0] != shuffled[1]
    assert all(sorted(x) == list(range(0, 12, 2)) for x in shuffled)


def data_types(root, cpu):
    for dtype in ("float32", ">f4", "uint8", "uint32", "int16", "int32"):
        create_fixture(root, count=2, dtype=dtype)
        assert run_reader(root, cpu) == [0, 1]
    for dtype in ("int8", "uint16", "int64", "float64"):
        create_fixture(root, count=2, dtype=dtype)
        rejected(lambda: make_pipeline(root, cpu), "data type")


def schema(root, cpu):
    import rocal_pybind as native

    assert not native.hdf5Reader(
        None,
        str(root),
        ["data"],
        [types.NONE],
        [],
        False,
        False,
        0,
        1,
        0,
        native.RocalShardingInfo(types.LAST_BATCH_FILL, False, True, -1),
    )
    checkpoint_pipe = Pipeline(
        batch_size=2, num_threads=1, rocal_cpu=cpu, enable_checkpointing=True
    )
    try:
        with checkpoint_pipe:
            rejected(
                lambda: fn.readers.hdf5(file_root=str(root), dataset_keys=["data"]),
                "checkpointing",
            )
    finally:
        checkpoint_pipe.rocal_release()
    create_fixture(root, count=2)
    rejected(lambda: make_pipeline(root, cpu, keys=["absent"]), "dataset key")
    rejected(lambda: make_pipeline(root, cpu, keys=["data", "data"]), "duplicate")
    rejected(
        lambda: make_pipeline(root, cpu, output_layouts=[types.NHW, types.NONE]),
        "layout",
    )
    for shape, fragment in (
        ((), "non-scalar"),
        ((0,), "dimension"),
        ((2**32, 1), "dimension"),
        ((2**31, 2**31), "overflows"),
        ((2,) * 5, "rank"),
    ):
        for f in root.glob("*.h5"):
            f.unlink()
        with h5py.File(root / "sample-00.h5", "w") as f:
            f.create_dataset(
                "data", shape=shape, dtype="float32", chunks=True if shape else None
            )
        rejected(lambda: make_pipeline(root, cpu, keys=["data"]), fragment)


def read_errors(root, cpu):
    for mode in ("missing", "key", "shape", "dtype", "corrupt"):
        create_fixture(root, count=32)
        pipe = make_pipeline(root, cpu, last_batch_policy=types.LAST_BATCH_DROP)
        late = root / "sample-31.h5"
        if mode == "missing":
            late.unlink()
        elif mode == "corrupt":
            late.write_bytes(b"invalid hdf5 file")
        else:
            with h5py.File(late, "a") as f:
                del f["label"]
                if mode == "shape":
                    f["label"] = np.zeros((20, 30), dtype=np.int32)
                elif mode == "dtype":
                    f["label"] = np.zeros((2, 3), dtype=np.float32)
        pipe.build()
        try:
            rejected(lambda: consume(pipe, root), "sample-31")
            # Reset after a repaired input must clear the previous
            # worker/context error.
            create_fixture(root, count=32)
            assert pipe.rocal_reset_loaders() == 0
            assert consume(pipe, root) == list(range(32))
        finally:
            pipe.rocal_release()


def lifecycle(root, cpu):
    create_fixture(root, count=16)
    for mode in ("before-build", "full-queue", "one-batch", "reset"):
        for repeat in range(10):
            pipe = make_pipeline(root, cpu)
            try:
                if mode != "before-build":
                    pipe.build()
                if mode == "full-queue":
                    time.sleep(0.01)
                if mode in ("one-batch", "reset"):
                    for epoch in range(10 if mode == "reset" else 1):
                        output = read_batch(pipe)
                        assert (
                            output is not None
                            and np.all(output[1][0] == 0)
                            and np.all(output[1][1] == 1)
                        )
                        if mode == "reset":
                            assert pipe.rocal_reset_loaders() == 0
            finally:
                pipe.rocal_release()


def end_of_input(root, cpu):
    create_fixture(root, count=3)
    for policy, expected in (
        (types.LAST_BATCH_FILL, [0, 1, 2, 2]),
        (types.LAST_BATCH_DROP, [0, 1]),
        (types.LAST_BATCH_PARTIAL, [0, 1, 2]),
    ):
        pipe = make_pipeline(root, cpu, last_batch_policy=policy,
                             pad_last_batch=True)
        try:
            pipe.build()
            for epoch in range(20):
                if epoch % 2:
                    # Also cover EOF published before the consumer starts.
                    time.sleep(0.01)
                assert consume(pipe, root, partial=policy == types.LAST_BATCH_PARTIAL) == expected
                for _ in range(3):
                    assert pipe.rocal_run() != types.OK
                    assert pipe.get_remaining_images() == 0
                assert pipe.rocal_reset_loaders() == types.OK
        finally:
            pipe.rocal_release()


def pytorch(root, cpu):
    from amd.rocal.plugin.pytorch import ROCALNumpyIterator

    create_fixture(root, count=5)
    pipe = make_pipeline(root, cpu, last_batch_policy=types.LAST_BATCH_PARTIAL)
    pipe.build()
    iterator = ROCALNumpyIterator(pipe, device="cpu")
    try:
        for epoch in range(2):
            labels = []
            for data, label in iterator:
                labels.extend(label[:, 0, 0].tolist())
            assert labels == list(range(5)), labels
            iterator.reset()
    finally:
        del iterator
    create_fixture(root, count=7)
    pipe = make_pipeline(
        root,
        cpu,
        last_batch_policy=types.LAST_BATCH_PARTIAL,
        num_shards=2,
        shard_id=0,
        stick_to_shard=False,
    )
    pipe.build()
    iterator = ROCALNumpyIterator(pipe, device="cpu")
    try:
        for expected in ([0, 2, 4, 6], [1, 3, 5], [0, 2, 4, 6]):
            labels = []
            for data, label in iterator:
                labels.extend(label[:, 0, 0].tolist())
            assert labels == expected, labels
            iterator.reset()
    finally:
        del iterator


def disabled(root, cpu):
    create_fixture(root, count=2)
    rejected(lambda: make_pipeline(root, cpu), "without HDF5")


CASES = {
    f.__name__: f
    for f in (
        values,
        layouts,
        file_order,
        batches,
        shards,
        epochs,
        data_types,
        schema,
        read_errors,
        lifecycle,
        end_of_input,
        pytorch,
        disabled,
    )
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.case:
        with tempfile.TemporaryDirectory(prefix="rocal-hdf5-") as directory:
            CASES[args.case](Path(directory), args.device == "cpu")
        print("HDF5_PASS", args.device, args.case, flush=True)
    else:
        for case in CASES:
            if case == "disabled":
                continue
            if case == "pytorch" and importlib.util.find_spec("torch") is None:
                print(
                    "HDF5_SKIP pytorch: install PyTorch to check its iterator",
                    flush=True,
                )
                continue
            # Bound each native test; crashes and hangs must fail the parent
            # process.
            result = subprocess.run(
                [sys.executable, __file__, "--device", args.device, "--case", case],
                timeout=90,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            print(result.stdout, end="", flush=True)
            result.check_returncode()
            assert f"HDF5_PASS {args.device} {case}" in result.stdout, result.stdout
        print("HDF5_READER_PASS", args.device, flush=True)


if __name__ == "__main__":
    main()
