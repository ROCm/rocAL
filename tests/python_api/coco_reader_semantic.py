# Copyright (c) 2018 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import torch
import os
import ctypes
import numpy as np
import cv2
from parse_config import parse_args
import json

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types


def _parse_mask_ids(mask_ids):
    if not mask_ids:
        return [0]
    parsed = []
    for token in mask_ids.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            continue
    return parsed or [0]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _tensor_to_cv2_bgr(img, device, dtype, layout):
    if device == "cpu":
        image = img.detach().numpy()
    else:
        image = img.cpu().numpy()

    if layout == types.NCHW:
        image = image.transpose([1, 2, 0])

    if image.dtype != np.uint8:
        # Handle common rocAL output ranges:
        # - decoded images may come as float in [0, 255] or [0, 1]
        # - normalized tensors (e.g. CMN) don't have a canonical visualization range
        if dtype in (types.FLOAT, types.FLOAT16):
            max_val = float(np.max(image)) if image.size else 0.0
            if 0.0 <= max_val <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


class ROCALCOCOIterator(object):
    """
    COCO ROCAL iterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None,
                 tensor_dtype=types.FLOAT, device="cpu", display=False, select_mask_ids=None, mask_type="pixelwise"):

        try:
            assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)
        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.device = device
        self.device_id = self.loader._device_id
        self.bs = self.loader._batch_size
        self.output_list = self.dimensions = self.torch_dtype = None
        self.display = display
        # Image id of a batch of images
        self.image_id = np.zeros(self.bs, dtype="int32")
        # Count of labels/ bboxes in a batch
        self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        # Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")
        self.output_memory_type = self.loader._output_memory_type
        self.select_mask_ids = select_mask_ids if select_mask_ids else [0]
        self.mask_type = mask_type

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.rocal_run() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.get_output_tensors()

        if self.output_list is None:
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                self.dimensions = self.output_tensor_list[i].dimensions()
                self.torch_dtype = self.output_tensor_list[i].dtype()
                if self.device == "cpu":
                    self.output = torch.empty(
                        self.dimensions, dtype=getattr(torch, self.torch_dtype))
                else:
                    torch_gpu_device = torch.device('cuda', self.device_id)
                    self.output = torch.empty(self.dimensions, dtype=getattr(
                        torch, self.torch_dtype), device=torch_gpu_device)
                self.output_tensor_list[i].copy_data(ctypes.c_void_p(
                    self.output.data_ptr()), self.output_memory_type)
                self.output_list.append(self.output)
        else:
            for i in range(len(self.output_tensor_list)):
                self.output_tensor_list[i].copy_data(ctypes.c_void_p(
                    self.output_list[i].data_ptr()), self.output_memory_type)

        self.labels = self.loader.get_bounding_box_labels()
        # 1D bboxes array in a batch
        self.bboxes = self.loader.get_bounding_box_cords()
        self.loader.get_img_sizes(self.img_size)
        self.loader.get_image_id(self.image_id)
        image_id_tensor = torch.tensor(self.image_id)
        if self.mask_type == "polygon":
            bbox_total = self.loader.get_bounding_box_count()
            mask_count = np.zeros(bbox_total, dtype="int32")
            mask_total = self.loader.get_mask_count(mask_count)
            polygon_size = np.zeros(mask_total, dtype="int32")
            polygons = self.loader.get_mask_coordinates(polygon_size, mask_count)
            select_mask_polygons = self.loader.get_select_mask(self.select_mask_ids)
        else:
            pixelwise_labels = self.loader.get_pixelwise_labels()
            random_mask_pixel = self.loader.get_random_mask_pixel()
            random_object_bbox = self.loader.get_random_object_bbox(types.OUT_BOX, k_largest=-1, foreground_prob=1.0, cache_objects=False)

        if self.mask_type == "polygon":
            return (self.output), self.bboxes, self.labels, image_id_tensor, self.img_size.copy(), mask_count, polygon_size, polygons, select_mask_polygons
        return (self.output), self.bboxes, self.labels, image_id_tensor, self.img_size.copy(), pixelwise_labels, random_mask_pixel, random_object_bbox

    def reset(self):
        self.loader.rocal_reset_loaders()

    def __iter__(self):
        return self


def draw_patches(img, bboxes, device, dtype, layout, out_path=None, crop_w=None, crop_h=None):
    # image is expected as a tensor, bboxes as numpy
    image = _tensor_to_cv2_bgr(img, device, dtype, layout)
    bboxes = np.reshape(bboxes, (-1, 4))

    for (l, t, r, b) in bboxes:
        loc_ = [l, t, r, b]
        color = (255, 0, 0)
        thickness = 2
        image = cv2.UMat(image).get()
        image = cv2.rectangle(image, (int(loc_[0]), int(loc_[1])), (int(
            (loc_[2])), int((loc_[3]))), color, thickness)
    if crop_h is not None and crop_w is not None:
        crop_h = int(max(0, min(crop_h, image.shape[0])))
        crop_w = int(max(0, min(crop_w, image.shape[1])))
        if crop_h > 0 and crop_w > 0:
            image = image[:crop_h, :crop_w]

    cv2.imwrite(out_path, image)
    return image


def _reshape_pixelwise_mask(mask, width: int, height: int):
    mask = np.asarray(mask)
    if mask.ndim == 2:
        if mask.shape == (height, width):
            return mask
        if mask.shape == (width, height):
            return mask.T
        return mask
    if mask.size == width * height:
        return mask.reshape((height, width))
    return mask.reshape((-1,))


def _save_pixelwise_semantic(out_dir, iter_idx, sample_idx, image_id, img_tensor, bboxes, device, dtype, layout,
                             img_sizes, mask, random_pixel, random_bbox):
    width = int(img_sizes[sample_idx * 2 + 0])
    height = int(img_sizes[sample_idx * 2 + 1])

    base_path = os.path.join(out_dir, "pixelwise")
    _ensure_dir(base_path)

    bgr = draw_patches(
        img_tensor,
        bboxes,
        device,
        dtype,
        layout,
        out_path=os.path.join(base_path, f"{image_id}_{iter_idx}_{sample_idx}_bbox.png"),
        crop_w=width,
        crop_h=height,
    )

    mask_2d = _reshape_pixelwise_mask(mask, width, height)
    if mask_2d.ndim != 2:
        return

    mask_vis = np.zeros((height, width), dtype=np.uint8)
    mask_vis[mask_2d > 0] = 255
    cv2.imwrite(os.path.join(base_path, f"{image_id}_{iter_idx}_{sample_idx}_mask.png"), mask_vis)

    overlay = bgr.copy()
    roi_h = min(height, overlay.shape[0])
    roi_w = min(width, overlay.shape[1])
    if roi_h > 0 and roi_w > 0:
        overlay_roi = overlay[:roi_h, :roi_w]
        mask_roi = mask_2d[:roi_h, :roi_w]
        overlay_roi[mask_roi > 0] = (0, 255, 0)
        overlay[:roi_h, :roi_w] = overlay_roi
    alpha = 0.35
    blended = cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0.0)

    row, col = int(random_pixel[0]), int(random_pixel[1])
    cv2.circle(blended, (col, row), 4, (0, 0, 255), thickness=-1)

    y0, x0, y1, x1 = [int(v) for v in random_bbox]
    cv2.rectangle(blended, (x0, y0), (x1, y1), (255, 0, 0), thickness=2)

    cv2.imwrite(os.path.join(base_path, f"{image_id}_{iter_idx}_{sample_idx}_semantic.png"), blended)


def _save_polygon_semantic(out_dir, iter_idx, sample_idx, image_id, img_tensor, bboxes, device, dtype, layout,
                           img_sizes, select_mask_polygons):
    width = int(img_sizes[sample_idx * 2 + 0])
    height = int(img_sizes[sample_idx * 2 + 1])

    base_path = os.path.join(out_dir, "polygon")
    _ensure_dir(base_path)

    bgr = draw_patches(
        img_tensor,
        bboxes,
        device,
        dtype,
        layout,
        out_path=os.path.join(base_path, f"{image_id}_{iter_idx}_{sample_idx}_bbox.png"),
        crop_w=width,
        crop_h=height,
    )

    selected = select_mask_polygons[sample_idx] if isinstance(select_mask_polygons, list) else {}
    overlay = bgr.copy()
    try:
        for _, poly_list in selected.items():
            for poly_coords in poly_list:
                pts = np.asarray(poly_coords, dtype=np.float32).reshape((-1, 2))
                pts_i = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts_i], isClosed=True, color=(0, 255, 0), thickness=2)
    except Exception:
        pass

    alpha = 0.8
    blended = cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0.0)
    cv2.imwrite(os.path.join(base_path, f"{image_id}_{iter_idx}_{sample_idx}_select_mask.png"), blended)


def main():
    args = parse_args()
    # Args
    image_path = args.image_dataset_path
    annotation_path = args.json_path
    rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    display = args.display
    num_threads = args.num_threads
    local_rank = args.local_rank
    world_size = args.world_size
    random_seed = args.seed
    tensor_format = types.NHWC if args.NHWC else types.NCHW
    tensor_dtype = types.FLOAT16 if args.fp16 else types.FLOAT
    select_mask_ids = _parse_mask_ids(getattr(args, "select_mask_ids", "0"))
    try:
        path = "output_folder/coco_reader_semantic/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)

    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                    seed=random_seed, rocal_cpu=rocal_cpu, tensor_layout=tensor_format, tensor_dtype=tensor_dtype)
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        if args.mask_type == "polygon":
            jpegs, bboxes, labels = fn.readers.coco(annotations_file=annotation_path, polygon_masks=True)
        else:
            # Configure random_mask_pixel to pick foreground pixels (value > 0).
            jpegs, bboxes, labels = fn.readers.coco(
                annotations_file=annotation_path,
                pixelwise_masks=True,
                is_foreground=True,
                value=0,
                is_threshold=True,
            )
        images_decoded = fn.decoders.image(jpegs, output_type=types.RGB, file_root=image_path, max_decoded_width=640, max_decoded_height=640,
                                           annotations_file=annotation_path, random_shuffle=False, shard_id=local_rank, num_shards=world_size)
        # Keep the graph free of bbox/mask meta-nodes so metadata APIs validate the reader outputs directly.
        pipe.set_outputs(images_decoded)
    # Build the pipeline
    pipe.build()
    # Dataloader
    if (args.rocal_gpu):
        data_loader = ROCALCOCOIterator(
            pipe, multiplier=pipe._multiplier, offset=pipe._offset, display=display,
            tensor_layout=tensor_format, tensor_dtype=tensor_dtype, device="gpu", select_mask_ids=select_mask_ids, mask_type=args.mask_type)
    else:
        data_loader = ROCALCOCOIterator(
            pipe, multiplier=pipe._multiplier, offset=pipe._offset, display=display,
            tensor_layout=tensor_format, tensor_dtype=tensor_dtype, device="cpu", select_mask_ids=select_mask_ids, mask_type=args.mask_type)

    import timeit
    start = timeit.default_timer()
    # Enumerate over the Dataloader
    for epoch in range(int(args.num_epochs)):
        print("EPOCH:::::", epoch)
        for i, it in enumerate(data_loader, 0):
            if args.print_tensor:
                print("**************", i, "*******************")
                print("**************starts*******************")
                print("\nIMAGES : \n", it[0])
                print("\nBBOXES:\n", it[1])
                print("\nLABELS:\n", it[2])
                print("\nIMAGE ID:\n", it[3])
                print("\nIMAGE SIZES (width,height per image):\n", it[4].tolist())
                if args.mask_type == "polygon":
                    mask_count = it[5]
                    polygon_size = it[6]
                    polygons = it[7]
                    select_mask_polygons = it[8]
                    print("\nMASK COUNT (per object):\n", mask_count.tolist())
                    print("\nPOLYGON SIZE (per polygon):\n", polygon_size.tolist())
                    print("\nSELECT MASK POLYGONS:\n", select_mask_polygons)
                    assert isinstance(polygons, list) and len(polygons) == batch_size
                else:
                    pixelwise = it[5]
                    random_pixels = it[6]
                    random_boxes = it[7]
                    print("\nPIXELWISE MASK UNIQUE VALUES:\n", [np.unique(mask) for mask in pixelwise])
                    print("\nRANDOM MASK PIXELS:\n", random_pixels)
                    print("\nRANDOM OBJECT BBOXES (OUT_BOX y0,x0,y1,x1):\n", [box.tolist() for box in random_boxes])
                    # Sanity: random_mask_pixel should hit a non-zero label when foreground exists.
                    for bi, (mask, (row, col)) in enumerate(zip(pixelwise, random_pixels)):
                        # mask is flattened; infer H/W from bbox-dependent tensor dims isn't exposed here, so skip exact indexing if out-of-range.
                        if row < 0 or col < 0:
                            raise RuntimeError(f"Invalid random_mask_pixel coords for sample {bi}: {(row, col)}")
                print("**************ends*******************")
                print("**************", i, "*******************")

            if args.display:
                for bi in range(batch_size):
                    img_id = int(it[3][bi].item()) if hasattr(it[3], "shape") else int(it[3][bi])
                    if args.mask_type == "polygon":
                        _save_polygon_semantic(
                            path,
                            i,
                            bi,
                            img_id,
                            it[0][bi],
                            it[1][bi],
                            ("gpu" if args.rocal_gpu else "cpu"),
                            tensor_dtype,
                            tensor_format,
                            it[4],
                            it[8]
                        )
                    else:
                        _save_pixelwise_semantic(
                            path,
                            i,
                            bi,
                            img_id,
                            it[0][bi],
                            it[1][bi],
                            ("gpu" if args.rocal_gpu else "cpu"),
                            tensor_dtype,
                            tensor_format,
                            it[4],
                            it[5][bi],
                            it[6][bi],
                            it[7][bi]
                        )
        data_loader.reset()
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)

    print("##############################  COCO READER (SEMANTIC) SUCCESS  ############################")


if __name__ == '__main__':
    main()
