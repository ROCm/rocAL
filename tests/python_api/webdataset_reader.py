
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
import torch
np.set_printoptions(threshold=1000, edgeitems=10000)
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import math
# import rocal_pybind.tensor
import sys
import cv2
import matplotlib.pyplot as plt
import os

def main():
    if  len(sys.argv) < 3:
        print ('Please pass tar_file index_file cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "audio"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    wds_data = sys.argv[1]
    index_file = sys.argv[2]
    if(sys.argv[3] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=300
    local_rank = 0
    world_size = 1
    print("*********************************************************************")
    webdataset_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)
    with webdataset_pipeline:
        img_raw = fn.readers.webdataset(
        path=wds_data, ext=[{'jpg', 'cls'}],
        )
        img = fn.decoders.webdataset(img_raw, file_root=wds_data, color_format=types.RGB)
        # resized = fn.resize(img, resize_shorter=256.0)
        resize_w = 400
        resize_h = 400
        scaling_mode = types.SCALING_MODE_STRETCH
        interpolation_type = types.LINEAR_INTERPOLATION
        if (scaling_mode == types.SCALING_MODE_STRETCH):
            resize_h = 416
        resized = fn.resize(img,
                               resize_width=resize_w,
                               resize_height=resize_h,
                               scaling_mode=scaling_mode,
                               interpolation_type=interpolation_type)
        tensor_format = types.NHWC
        tensor_dtype = types.FLOAT
        output = fn.crop_mirror_normalize(resized,
                                              crop=(224, 224),
                                              crop_pos_x=0.0,
                                              crop_pos_y=0.0,
                                              mean=[0, 0, 0],
                                              std=[1, 1, 1],
                                              mirror=0,
                                              output_layout=tensor_format,
                                              output_dtype=tensor_dtype)

        webdataset_pipeline.set_outputs(output)
    webdataset_pipeline.build()
    audioIteratorPipeline = ROCALClassificationIterator(webdataset_pipeline, auto_reset=True)
    cnt = 0
    for e in range(1):
        print("Epoch :: ", e)
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i , it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************",i)
            for img, label in zip(it[0],it[1]):
                print("label", label)
                print("img",img)
        print("EPOCH DONE", e)
if __name__ == '__main__':
    main()
