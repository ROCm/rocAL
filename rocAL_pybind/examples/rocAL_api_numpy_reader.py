from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.pytorch import ROCALNumpyIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import os
import numpy as np

def main():
    if  len(sys.argv) < 3:
        print ('Please pass numpy_folder cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/NUMPY_READER/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    if(sys.argv[2] == "cpu"):
        rocal_cpu = True
    else:
        rocal_cpu = False
    batch_size = int(sys.argv[3])
    num_threads = 1
    device_id = 0
    local_rank = 0
    world_size = 1
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)

    files_list = []
    for file in os.listdir(data_path):
        files_list.append(os.path.join(data_path, file))

    import time
    start = time.time()
    pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu)

    with pipeline:
        numpy_reader_output = fn.readers.numpy(file_root=data_path, shard_id=local_rank, num_shards=world_size)
        new_output = fn.set_layout(numpy_reader_output, output_layout=types.NCDHW)
        brightness_output = fn.brightness(new_output, brightness=1.25, brightness_shift=0.0, output_layout=types.NCDHW, output_dtype=types.FLOAT)
        pipeline.set_outputs(brightness_output)

    pipeline.build()
    
    numpyIteratorPipeline = ROCALNumpyIterator(pipeline, tensor_dtype=types.UINT8)
    print(len(numpyIteratorPipeline))
    cnt = 0
    for epoch in range(1):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , [it] in enumerate(numpyIteratorPipeline):
            print(i, it.shape)
            for j in range(batch_size):
                arr = np.load(files_list[cnt])
                shape = arr.shape
                print(np.array_equal(arr * 1.25, it[j].cpu().numpy()[:, :shape[1], :shape[2], :shape[3]]))
                cnt += 1
            print("************************************** i *************************************",i)
        numpyIteratorPipeline.reset()
    print("*********************************************************************")
    print(f'Took {time.time() - start} seconds')

if __name__ == '__main__':
    main()
