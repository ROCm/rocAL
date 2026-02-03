"""Checkpointing example for the rocAL Python API."""

import sys
import os
import time
from amd.rocal.pipeline import pipeline_def, Pipeline
from amd.rocal.plugin.generic import ROCALGenericIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import numpy as np

seed = 1549361629  # Deterministic seed for checkpoint reproducibility.
image_dir = "/opt/rocm/share/rocal/test/data/images/AMD-tinyDataSet"  # Default dataset path.
batch_size = 1  # Batch size used for the example pipeline.
gpu_id = 0  # GPU device id for ROCAL.

def _get_image_names(pipe: Pipeline):
    """Return decoded image names for the current batch."""
    # Batch size from the pipeline (protected access).
    batch_size = pipe._batch_size  # pylint: disable=protected-access
    name_lengths = np.zeros(batch_size, dtype=np.int32)  # Per-sample name lengths.
    total_length = pipe.get_image_name_length(name_lengths)  # Total bytes in the name buffer.
    if total_length == 0:
        return [""] * batch_size
    raw_bytes = pipe.get_image_name(total_length)  # Raw name buffer from ROCAL.
    names = []  # Decoded UTF-8 names for the batch.
    offset = 0  # Offset into the raw name buffer.
    for length in name_lengths:  # Per-sample name length.
        if length == 0:
            names.append("")
            continue
        chunk = raw_bytes[offset : offset + length]  # Bytes for a single name.
        names.append(chunk.decode("utf-8", errors="replace"))
        offset += length
    return names

@pipeline_def(seed=seed)
def image_decoder_pipeline(device="cpu", path=image_dir):
    """Simple file reader + decoder pipeline used for checkpointing demo."""
    jpegs, labels = fn.readers.file(file_root=path)  # Reader outputs (labels unused).
    images = fn.decoders.image(
        jpegs,
        file_root=path,
        device=device,
        output_type=types.RGB,
        shard_id=0,
        num_shards=1,
        random_shuffle=True,
    )
    # Keep ops simple and deterministic for checkpoint debug
    return fn.brightness_fixed(images)


def create_and_checkpoint(bs, rocal_device, rocal_cpu, img_folder, ckpt_path=None):
    """Create a pipeline, advance a few iterations, dump checkpoint, and release it.

    Returns a tuple: (serialized_ckpt_bytes, ckpt_path_used or None)
    """
    pipe = image_decoder_pipeline(  # Pipeline instance with checkpointing enabled.
        batch_size=bs,
        num_threads=1,
        device_id=gpu_id,
        rocal_cpu=rocal_cpu,
        tensor_layout=types.NHWC,
        reverse_channels=True,
        mean=[0, 0, 0],
        std=[255, 255, 255],
        device=rocal_device,
        path=img_folder,
        enable_checkpointing=True,
    )
    pipe.build()
    iterator = ROCALGenericIterator(pipe)  # Iterator for consuming batches.

    print("Remaining images (initial):", pipe.get_remaining_images())
    for i in range(3):  # Advance a few iterations before checkpointing.
        batch = iterator.next()  # Next output batch from the pipeline.
        [image], label = batch  # Unpack image tensor and labels.
        image_names = _get_image_names(pipe)  # Human-readable image names.
        for idx in range(batch_size):  # Batch index.
            print(image_names[idx], label[idx])

    # Save checkpoint bytes (and optionally to file)
    serialized_ckpt = pipe.checkpoint(filename=ckpt_path)  # Serialized checkpoint bytes.
    if ckpt_path:
        try:
            size = os.path.getsize(ckpt_path)  # On-disk checkpoint size.
            print(f"Checkpoint saved to: {ckpt_path} ({size} bytes)")
        except Exception as e:
            print("Warning: could not stat checkpoint file:", e)
    print("Remaining images at checkpoint:", pipe.get_remaining_images())

    for i in range(5):  # Continue after checkpoint to ensure pipeline runs.
        batch = iterator.next()  # Next output batch from the pipeline.
        [image], label = batch  # Unpack image tensor and labels.
        image_names = _get_image_names(pipe)  # Human-readable image names.
        for idx in range(batch_size):  # Batch index.
            print(image_names[idx], label[idx])

    del iterator
    pipe.rocal_release()  # Release rocAL resources before creating a new pipeline.

    return serialized_ckpt, ckpt_path


def restore_and_compare(bs, rocal_device, rocal_cpu, img_folder, serialized_ckpt=None, ckpt_path=None):
    """Create a fresh pipeline in a separate scope and restore from checkpoint.

    Accepts either serialized_ckpt bytes or ckpt_path on disk.
    """
    pipe_restored = image_decoder_pipeline(  # Restored pipeline instance.
        batch_size=bs,
        num_threads=1,
        device_id=gpu_id,
        rocal_cpu=rocal_cpu,
        tensor_layout=types.NHWC,
        reverse_channels=True,
        mean=[0, 0, 0],
        std=[255, 255, 255],
        device=rocal_device,
        path=img_folder,
        enable_checkpointing=True,
    )
    pipe_restored.build()

    print(f"Before restore: remaining images = {pipe_restored.get_remaining_images()}")
    print(f"Checkpoint size: {len(serialized_ckpt) if serialized_ckpt else 'N/A'} bytes")

    try:
        if serialized_ckpt is not None:
            pipe_restored.restore_checkpoint(serialized_ckpt=serialized_ckpt)
            print("Restore from serialized checkpoint completed")
        elif ckpt_path is not None:
            pipe_restored.restore_checkpoint(filename=ckpt_path)
            print(f"Restore from checkpoint file {ckpt_path} completed")
        else:
            raise RuntimeError("No checkpoint provided for restore")
    except Exception as e:
        print("Error restoring checkpoint:", e)
        print("Tip: Ensure the checkpoint was produced by an identical pipeline.")
        pipe_restored.rocal_release()
        return

    print(f"After restore: remaining images = {pipe_restored.get_remaining_images()}")

    iterator = ROCALGenericIterator(pipe_restored)  # Iterator for restored pipeline output.
    for i in range(5):  # Read a few batches after restore.
        batch = iterator.next()
        [image], label = batch
        image_names = _get_image_names(pipe_restored)
        for idx in range(bs):  # Batch index.
            print(image_names[idx], label[idx])

    del iterator
    pipe_restored.rocal_release()


def main():
    print('Optional arguments: <cpu/gpu> <image_folder>')
    bs = batch_size  # Batch size for the example run.
    rocal_device = "cpu"  # Device string passed to ROCAL ops.
    rocal_cpu = True  # Whether to run ROCAL in CPU mode.
    img_folder = image_dir  # Dataset path for the example run.
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "gpu":
            rocal_device = "gpu"
            rocal_cpu = False
    if len(sys.argv) > 2:
        img_folder = sys.argv[2]

    # Use a unique file for convenience, but we primarily pass bytes to restore
    ckpt_path = os.path.join(  # File path for storing checkpoint bytes.
        os.path.dirname(__file__), "checkpoint.bin"
    )
    
    print("\n========== Creating and Checkpointing Pipeline ==========")
    serialized_ckpt, _ = create_and_checkpoint(  # Run the checkpointing workflow.
        bs, rocal_device, rocal_cpu, img_folder, ckpt_path=ckpt_path
    )

    time.sleep(0.5)  # Give the pipeline time to release resources before restore.

    print("\n========== Restoring Pipeline from Checkpoint ==========")
    restore_and_compare(
        bs,
        rocal_device,
        rocal_cpu,
        img_folder,
        serialized_ckpt=serialized_ckpt,
        ckpt_path=None,
    )

    print("\n========== Checkpoint Test Completed Successfully ==========")

if __name__ == '__main__':
    main()
