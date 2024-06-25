
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
    import cv2
    if device == "cpu":
        img = img.detach().numpy()
    else:
        img = img.cpu().numpy()
    if dtype == types.FLOAT16:
        img = (img).astype('uint8')

    if color_format == types.RGB:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("output_folder/webdataset_reader/" + str(idx)+"_"+"train" + ".png", img,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])

def main():
    if  len(sys.argv) < 3:
        print ('Please pass tar_file index_file cpu/gpu batch_size. If no index file is present, please pass empty string.')
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
    color_format=types.RGB
    print("*********************************************************************")
    webdataset_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu, tensor_dtype = types.UINT8, )
    with webdataset_pipeline:
        img_raw = fn.readers.webdataset(
        path=wds_data, ext=[{'jpg', 'json', 'txt'}], 
        index_paths = index_file)
        img = fn.decoders.webdataset(img_raw, file_root=wds_data, 
                                     index_path = index_file, 
                                     color_format=color_format,max_decoded_width=500, max_decoded_height=500)
        tensor_dtype = types.UINT8

        webdataset_pipeline.set_outputs(img)
    webdataset_pipeline.build()
    audioIteratorPipeline = ROCALClassificationIterator(webdataset_pipeline, auto_reset=True)
    cnt = 0
    for e in range(1):
        print("Epoch :: ", e)
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i, it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************",i)
            print("length of image data: ", len(it[0]))
            for x in range(len(it[0])):
                for img, label in zip(it[0][x], it[1]):
                    print("img data", img)
                    print("label ascii data", label)
                    draw_patches(img, cnt, "cpu", tensor_dtype, color_format=color_format)
                    cnt = cnt + 1
            break
                
        print("EPOCH DONE", e)

if __name__ == '__main__':
    main()
