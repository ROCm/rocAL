from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.generic import ROCALNumpyIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import os


def main():
    if len(sys.argv) < 3:
        print('Please pass numpy_folder cpu/gpu batch_size')
        exit(0)
    data_path = sys.argv[1]
    if (sys.argv[2] == "cpu"):
        rocal_cpu = True
    else:
        rocal_cpu = False
    batch_size = int(sys.argv[3])
    num_threads = 1
    device_id = 0
    local_rank = 0
    world_size = 1
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)

    pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu)

    with pipeline:
        numpy_reader_output = fn.readers.numpy(file_root=data_path, shard_id=local_rank, num_shards=world_size)
        pipeline.set_outputs(numpy_reader_output)

    pipeline.build()

    numpyIteratorPipeline = ROCALNumpyIterator(pipeline, return_max_roi=True)
    print(len(numpyIteratorPipeline))
    for epoch in range(1):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , [it] in enumerate(numpyIteratorPipeline):
            print(it.shape)
            print("************************************** i *************************************",i)
        numpyIteratorPipeline.reset()
    print("*********************************************************************")


if __name__ == '__main__':
    main()
