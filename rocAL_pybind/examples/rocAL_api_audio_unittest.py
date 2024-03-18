
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
from amd.rocal.plugin.pytorch import ROCALAudioIterator
import torch
# torch.set_printoptions(threshold=10_000)
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
def draw_patches(img, idx, device):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    audio_data = image.flatten()
    label = idx
    print("label: ", label)
    # Saving the array in a text file
    file = open("results/rocal_data_new"+str(label)+".txt", "w+")
    content = str(audio_data)
    file.write(content)
    file.close()
    plt.plot(audio_data)
    plt.savefig("results/rocal_data_new"+str(label)+".png")
    plt.close()
def main():
    if  len(sys.argv) < 3:
        print ('Please pass audio_folder file_list cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "audio"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    file_list = " "
    _rocal_cpu = True # The GPU support for Audio is not given yet
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    print("*********************************************************************")
    audio_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rocal_cpu)
    with audio_pipeline:
        audio_decode = fn.decoders.audio(file_root=data_path, file_list_path=file_list, downmix=False, shard_id=0, num_shards=2, storage_type=0, stick_to_shard=False)
        audio_pipeline.set_outputs(audio_decode)
    audio_pipeline.build()
    audioIteratorPipeline = ROCALAudioIterator(audio_pipeline, auto_reset=True)
    cnt = 0
    for e in range(1):
        print("Epoch :: ", e)
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i , it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************",i)
            for x in range(len(it[0])):
                for img, label in zip(it[0][x], it[1]):
                    print("label", label)
                    print("cnt", cnt)
                    print("img", img)
                    draw_patches(img, cnt, "cpu")
                    cnt+=1
        print("EPOCH DONE", e)
if __name__ == '__main__':
    main()

