from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.generic import ROCALNumpyIterator
import amd.rocal.fn as fn
import sys
import os

def draw_patches(image, idx, layout="nhwc", dtype="uint8"):
    # image is expected as a numpy array
    import cv2
    if layout == "nchw":
        image = image.transpose([1, 2, 0])
    if dtype == "fp16":
        image = image.astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_folder/numpy_reader/" + str(idx) + ".png", image)

def main():
    if len(sys.argv) < 3:
        print('Please pass numpy_folder cpu/gpu batch_size')
        exit(0)
    data_path = sys.argv[1]
    try:
        path = "output_folder/numpy_reader/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
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

    cnt = 0
    numpyIteratorPipeline = ROCALNumpyIterator(pipeline)
    print(len(numpyIteratorPipeline))
    for epoch in range(1):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , [batch] in enumerate(numpyIteratorPipeline):
            print(batch.shape)
            for img in batch:
                draw_patches(img, cnt)
                cnt += 1
        numpyIteratorPipeline.reset()
    print("*********************************************************************")


if __name__ == '__main__':
    main()
