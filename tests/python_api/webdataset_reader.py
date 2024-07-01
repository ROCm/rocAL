
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

def draw_patches(img, idx, device, dtype, color_format=types.RGB):
    # image is expected as a tensor, bboxes as numpy
    import cv2
    if device == "cpu":
        img = img.detach().numpy()
    else:
        img = img.cpu().numpy()
    if dtype == types.FLOAT16:
        img = (img).astype('uint8')
    print("img shape", img.shape)

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
        path=wds_data, ext=[{'jpg', 'cls'}],
        )
        img = fn.decoders.webdataset(img_raw, file_root=wds_data, color_format=color_format,max_decoded_width=500, max_decoded_height=500)

        tensor_format = types.NHWC
        tensor_dtype = types.FLOAT


        webdataset_pipeline.set_outputs(img)
    webdataset_pipeline.build()
    audioIteratorPipeline = ROCALClassificationIterator(webdataset_pipeline, auto_reset=True)
    cnt = 0
    for epoch in range(1):
        print("EPOCH:::::", epoch)
        for i, (output_list, labels) in enumerate(audioIteratorPipeline, 0):
            for j in range(len(output_list)):
                print("**************", i, "*******************")
                print("**************starts*******************")
                print("\nImages:\n", output_list[j])
                print("\nLABELS:\n", labels)
                print("**************ends*******************")
                print("**************", i, "*******************")
                for img in output_list[j]:
                    draw_patches(img, cnt, "cpu", tensor_dtype, color_format=color_format)
                    cnt += 1

        audioIteratorPipeline.reset()
                
        print("EPOCH DONE")

if __name__ == '__main__':
    main()
