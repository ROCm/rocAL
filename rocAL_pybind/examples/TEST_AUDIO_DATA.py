
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

def dump_bin_file(audio_data, filename, augmentation_name):
    # Ensure audio_data is a NumPy array
    audio_data = np.array(audio_data)
    path = filename 
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    # Open the binary file in write mode
    print(filename)
    filename = os.path.join(path, augmentation_name)
    filename = filename + ".bin"
    with open(filename, 'wb') as file:
        # Write the audio data to the binary file
        audio_data.tofile(file)

def main():
    if  len(sys.argv) < 3:
        print ('Please pass audio_folder file_list')
        exit(0)
    try:
        path= "OUTPUTS_AUDIO/BIN_OUTPUTS/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    file_list = sys.argv[2]
    _rocal_cpu = True
    batch_size = 3
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    print("*********************************************************************")
    audio_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rocal_cpu)
    
    def pre_emphasis_filter():
        augmentation_name = "pre_emphasis_filter"
        with audio_pipeline:
            audio_decode = fn.decoders.audio(file_root=data_path, file_list_path="", downmix=False, shard_id=0, num_shards=2, stick_to_shard=False)
            pre_emphasis_filter = fn.preemphasis_filter(audio_decode)
            audio_pipeline.set_outputs(pre_emphasis_filter)
        audio_pipeline.build()
        return audio_pipeline, augmentation_name
    
    # Test Case 1 - PreEmphasis filter
    pre_emphasis_filter_pipeline, augmentation_name = pre_emphasis_filter()
    audioIteratorPipeline = ROCALAudioIterator(pre_emphasis_filter_pipeline, auto_reset=True)
    
    cnt = 0

    torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
    for i , it in enumerate(audioIteratorPipeline):
        print("************************************** i *************************************",i)
        for x in range(len(it[0])):
            for audio_data, _ in zip(it[0][x], it[1]):
                print("cnt", cnt)
                print("audio_data", audio_data)
                dump_bin_file(audio_data, filename = os.path.join(path, augmentation_name), augmentation_name=augmentation_name)
                cnt+=1

if __name__ == '__main__':
    main()

