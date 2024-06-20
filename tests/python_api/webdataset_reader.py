
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
import sys
import matplotlib.pyplot as plt
import os
import cv2

def draw_patches(img, idx, device, dtype, layout, color_format=types.RGB):
    # image is expected as a tensor, bboxes as numpy
    import cv2
    if device == "cpu":
        img = img.detach().numpy()
    else:
        img = img.cpu().numpy()
    if dtype == types.FLOAT16:
        img = (img).astype('uint8')
    print("img shapeeeeee", img.shape)

    if color_format == types.RGB:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("output_folder/webdataset_reader/" + str(idx)+"_"+"train" + ".png", img,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])

def main():
    if  len(sys.argv) < 3:
        print ('Please pass tar_file index_file cpu/gpu batch_size')
        exit(0)
    try:
        path= "output_folder/webdataset_reader/"
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
    color_format=types.RGB
    print("*********************************************************************")
    webdataset_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu, tensor_dtype = types.UINT8, )
    with webdataset_pipeline:
        img_raw = fn.readers.webdataset(
        path=wds_data, ext=[{'jpg', 'cls'}], index_paths = index_file,
        )
        img = fn.decoders.webdataset(img_raw, file_root=wds_data, index_path = index_file, color_format=color_format,max_decoded_width=500, max_decoded_height=500)
        # resized = fn.resize(img, resize_shorter=256.0)
        # resize_w = 400
        # resize_h = 400
        # scaling_mode = types.SCALING_MODE_STRETCH
        # interpolation_type = types.LINEAR_INTERPOLATION
        # if (scaling_mode == types.SCALING_MODE_STRETCH):
        #     resize_h = 416
        # resized = fn.resize(img,
        #                        resize_width=resize_w,
        #                        resize_height=resize_h,
        #                        scaling_mode=scaling_mode,
        #                        interpolation_type=interpolation_type)
        tensor_format = types.NHWC
        tensor_dtype = types.FLOAT
        # output = fn.crop_mirror_normalize(resized,
        #                                       crop=(224, 224),
        #                                       crop_pos_x=0.0,
        #                                       crop_pos_y=0.0,
        #                                       mean=[0, 0, 0],
        #                                       std=[1, 1, 1],
        #                                       mirror=0,
        #                                       output_layout=tensor_format,
        #                                       output_dtype=tensor_dtype)

        webdataset_pipeline.set_outputs(img)
    webdataset_pipeline.build()
    audioIteratorPipeline = ROCALClassificationIterator(webdataset_pipeline, auto_reset=True)
    cnt = 0
    for e in range(1):
        print("Epoch :: ", e)
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i , it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************",i)
            print(it[1])
            for x in range(len(it[0])):
                for img, label in zip(it[0][x], it[1]):
                    print("label", label)
                    print("label shape",label.shape)
                    draw_patches(img, cnt, "cpu", types.UINT8, tensor_format, color_format=color_format)
                    cnt = cnt + 1
            break
                
        print("EPOCH DONE", e)

if __name__ == '__main__':
    main()
