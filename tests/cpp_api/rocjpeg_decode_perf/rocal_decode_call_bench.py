import argparse
import contextlib
import io
import multiprocessing as mp
import os
import sys
import time

from amd.rocal.pipeline import pipeline_def
import amd.rocal.fn as fn
import amd.rocal.types as types


@pipeline_def(seed=1549361629)
def image_decoder_pipeline(device="cpu", path="", output_type="rgb", shard_id=0, num_shards=1):
    jpegs, labels = fn.readers.file(file_root=path)

    output_type_map = {
        "rgb": types.RGB,
        "gray": types.GRAY,
    }

    images = fn.decoders.image(
        jpegs,
        file_root=path,
        device=device,
        output_type=output_type_map[output_type],
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=False,
    )

    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple rocAL decode benchmark")
    parser.add_argument("--path", required=True, help="Input image directory")
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-type", choices=["rgb", "gray"], default="rgb")
    parser.add_argument(
        "--total-files-on-disk",
        type=int,
        default=-1,
        help="Optional known file count. If set, skip os.walk count.",
    )
    return parser.parse_args()


def split_path_enabled(args):
    split_mode = os.environ.get("ROCAL_ROCJPEG_DEDICATED_OMP_SPLIT", "")
    return (
        args.device == "gpu"
        and split_mode not in ("0", "OFF", "off", "FALSE", "false")
    )


def effective_batch_size(args):
    if not split_path_enabled(args):
        return args.batch_size

    rocjpeg_decoder_threads = max(1, min(4, args.num_threads))
    return args.batch_size * rocjpeg_decoder_threads


def is_jpeg_file(filename):
    return os.path.splitext(filename)[1].lower() in (".jpg", ".jpeg")


def count_jpeg_files(path):
    total = 0
    for _, _, files in os.walk(path, followlinks=True):
        total += sum(1 for filename in files if is_jpeg_file(filename))
    return total


def files_in_shard(total_files, shard_id, num_shards):
    if total_files < 0:
        return -1
    base = total_files // num_shards
    remainder = total_files % num_shards
    return base + (1 if shard_id < remainder else 0)


def extract_timing_info(pipe):
    info = pipe.timing_info()
    timing = {}

    for field in [
        "load_time",
        "decode_time",
        "process_time",
        "transfer_time",
    ]:
        if hasattr(info, field):
            timing[field] = getattr(info, field)

    return timing


def add_timing_info(total, batch_info):
    for key, value in batch_info.items():
        total[key] = total.get(key, 0) + value
    return total


def run_one_shard(args, shard_id, num_shards, device_id, total_files):
    rocal_batch_size = effective_batch_size(args)
    shard_file_count = files_in_shard(total_files, shard_id, num_shards)

    pipe = image_decoder_pipeline(
        batch_size=rocal_batch_size,
        num_threads=args.num_threads,
        device_id=device_id,
        rocal_cpu=(args.device == "cpu"),
        tensor_layout=types.NHWC,
        reverse_channels=True,
        mean=[0, 0, 0],
        std=[255, 255, 255],
        device=args.device,
        path=args.path,
        output_type=args.output_type,
        shard_id=shard_id,
        num_shards=num_shards,
    )

    pipe.build()

    print(f"Requested batch size: {args.batch_size}")
    if rocal_batch_size != args.batch_size:
        print(f"Effective rocAL batch size: {rocal_batch_size}")

    print(f"GPU/device id: {device_id}")
    print(f"Shard id: {shard_id}")
    print(f"Num shards: {num_shards}")
    print(f"Decoding started with {args.num_threads} threads, please wait!")
    sys.stdout.flush()

    accumulated_timing_info = {}

    start_time = time.perf_counter()

    while pipe.get_remaining_images() > 0:
        status = pipe.rocal_run()
        if not status:
            break

        accumulated_timing_info = add_timing_info(
            accumulated_timing_info,
            extract_timing_info(pipe),
        )

    total_elapsed_s = time.perf_counter() - start_time

    try:
        pipe.rocal_run()
    except Exception:
        pass

    accumulated_timing_info = add_timing_info(
        accumulated_timing_info,
        extract_timing_info(pipe),
    )

    decoded_images = shard_file_count if shard_file_count >= 0 else total_files

    avg_time_per_image_ms = 0.0
    images_per_sec = 0.0

    if decoded_images > 0 and total_elapsed_s > 0:
        avg_time_per_image_ms = (total_elapsed_s * 1000.0) / decoded_images
        images_per_sec = decoded_images / total_elapsed_s

    print(f"Total decoded images: {decoded_images}")
    print(f"Total processing time (sec): {total_elapsed_s:.6f}")
    print(f"Average processing time per image (ms): {avg_time_per_image_ms:.6f}")
    print(f"Average decoded images per sec (Images/Sec): {images_per_sec:.2f}")

    decode_time_us = accumulated_timing_info.get("decode_time", 0)
    load_time_us = accumulated_timing_info.get("load_time", 0)
    process_time_us = accumulated_timing_info.get("process_time", 0)

    if decode_time_us:
        print(f"rocAL internal decode time (sec): {decode_time_us / 1_000_000.0:.6f}")

    if load_time_us:
        print(f"rocAL internal load time (sec): {load_time_us / 1_000_000.0:.6f}")

    if process_time_us:
        print(f"rocAL internal process time (sec): {process_time_us / 1_000_000.0:.6f}")

    print("Decoding completed!")
    return {
        "device_id": device_id,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "decoded_images": decoded_images,
        "total_elapsed_s": total_elapsed_s,
        "decode_time_s": decode_time_us / 1_000_000.0 if decode_time_us else 0.0,
        "load_time_s": load_time_us / 1_000_000.0 if load_time_us else 0.0,
        "process_time_s": process_time_us / 1_000_000.0 if process_time_us else 0.0,
    }


def run_worker(args, shard_id, num_shards, device_id, total_files, queue):
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = run_one_shard(args, shard_id, num_shards, device_id, total_files)
        queue.put((shard_id, True, result, output.getvalue()))
    except Exception as exc:
        queue.put((shard_id, False, repr(exc), output.getvalue()))


def print_multi_gpu_summary(results):
    print("")
    print("Per-GPU/shard processing time:")
    for result in sorted(results, key=lambda item: item["shard_id"]):
        print(
            f"GPU/device {result['device_id']} shard {result['shard_id']} "
            f"total processing time (sec): {result['total_elapsed_s']:.6f}"
        )

    if results:
        avg_total = sum(item["total_elapsed_s"] for item in results) / len(results)
        avg_decode = sum(item["decode_time_s"] for item in results) / len(results)
        max_total = max(item["total_elapsed_s"] for item in results)
        print(f"Average total processing time across {len(results)} GPUs/shards (sec): {avg_total:.6f}")
        if avg_decode:
            print(f"Average rocAL internal decode time across {len(results)} GPUs/shards (sec): {avg_decode:.6f}")
        print(f"Wall-clock equivalent time across {len(results)} GPUs/shards (sec): {max_total:.6f}")


def main():
    args = parse_args()

    total_files = args.total_files_on_disk if args.total_files_on_disk >= 0 else count_jpeg_files(args.path)
    num_shards = max(1, args.num_shards)
    num_gpus = max(1, args.num_gpus)

    if num_shards == 1:
        run_one_shard(args, 0, 1, args.device_id, total_files)
        return

    mp_context = mp.get_context("spawn")
    workers = []
    queue = mp_context.Queue()
    for shard_id in range(num_shards):
        device_id = args.device_id + (shard_id % num_gpus)
        process = mp_context.Process(
            target=run_worker,
            args=(args, shard_id, num_shards, device_id, total_files, queue),
        )
        process.start()
        workers.append(process)

    worker_results = []
    failed = False
    for _ in workers:
        shard_id, ok, result, output = queue.get()
        print("")
        print(f"========== Worker output for shard {shard_id} ==========")
        print(output, end="")
        if ok:
            worker_results.append(result)
        else:
            failed = True
            print(f"ERROR: shard {shard_id} failed: {result}")

    for process in workers:
        process.join()
        if process.exitcode != 0:
            failed = True

    if failed:
        raise SystemExit(1)

    print_multi_gpu_summary(worker_results)


if __name__ == "__main__":
    main()
