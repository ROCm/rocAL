
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from amd.rocal.plugin.pytorch import ROCALAudioIterator
import torch
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
np.set_printoptions(threshold=1000, edgeitems=10000)

def plot_audio_wav(audio_tensor, idx):
    #image is expected as a tensor, bboxes as numpy
    audio_data = audio_tensor.detach().numpy()
    audio_data = audio_data.flatten()
    label = idx.cpu().detach().numpy()
    print("label: ", label)
    # Saving the array in a text file
    file = open("results/rocal_data_new" + str(label) + ".txt", "w+")
    content = str(audio_data)
    file.write(content)
    file.close()
    plt.plot(audio_data)
    plt.savefig("results/rocal_data_new" + str(label) + ".png")
    plt.close()

def main():
    if  len(sys.argv) < 3:
        print ('Please pass audio_folder batch_size')
        exit(0)
    try:
        path= "results"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    file_list = sys.argv[2]
    rocal_cpu = True # The GPU support for Audio is not given yet
    batch_size = int(sys.argv[3])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    print("*********************************************************************")
    audio_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu)
    with audio_pipeline:
        audio, label = fn.readers.file(file_root=data_path, file_list=file_list)
        audio_decode = fn.decoders.audio(audio, file_root=data_path, file_list_path=file_list, downmix=False, shard_id=0, num_shards=1, stick_to_shard=False)
        pre_emphasis_filter = fn.preemphasis_filter(audio_decode)
        audio_pipeline.set_outputs(pre_emphasis_filter)
    audio_pipeline.build()
    audioIteratorPipeline = ROCALAudioIterator(audio_pipeline, auto_reset=True)
    cnt = 0
    for epoch in range(1):
        print("Epoch :: ", epoch)
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i , it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************",i)
            for x in range(len(it[0])):
                for audio_tensor, label, roi in zip(it[0][x], it[1], it[2]):
                    print("label", label)
                    print("cnt", cnt)
                    print("Audio", audio_tensor)
                    print("Roi", roi)
                    plot_audio_wav(audio_tensor, label)
                    cnt+=1
        print("EPOCH DONE", epoch)
if __name__ == '__main__':
    main()

