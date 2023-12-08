from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.pytorch import ROCALNumpyIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import os, glob

val_cases_list = ['00000', '00003', '00005', '00006', '00012', '00024', '00034', '00041', '00044', '00049', '00052', '00056', '00061', '00065', '00066', '00070', '00076', '00078', '00080', '00084',
                  '00086', '00087', '00092', '00111', '00112', '00125', '00128', '00138', '00157', '00160', '00161', '00162', '00169', '00171', '00176', '00185', '00187', '00189', '00198', '00203', '00206', '00207']

def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data

def get_data_split(path: str):
    imgs = load_data(path, "*_x.npy")
    lbls = load_data(path, "*_y.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    imgs_train, lbls_train, imgs_val, lbls_val = [], [], [], []
    for (case_img, case_lbl) in zip(imgs, lbls):
        if case_img.split("_")[-2] in val_cases_list:
            imgs_val.append(case_img)
            lbls_val.append(case_lbl)
        else:
            imgs_train.append(case_img)
            lbls_train.append(case_lbl)

    return imgs_train, imgs_val, lbls_train, lbls_val

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
    num_threads = 8
    device_id = 0
    local_rank = 0
    world_size = 1
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    x_train, x_val, y_train, y_val = get_data_split(data_path)

    import time
    start = time.time()
    pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu, prefetch_queue_depth=2)

    with pipeline:
        numpy_reader_output = fn.readers.numpy(file_root=data_path, files=x_train, shard_id=local_rank, num_shards=world_size, random_shuffle=True, seed=random_seed+local_rank)
        numpy_reader_output1 = fn.readers.numpy(file_root=data_path, files=y_train, shard_id=local_rank, num_shards=world_size, random_shuffle=True, seed=random_seed+local_rank)
        data_output = fn.set_layout(numpy_reader_output, output_layout=types.NCDHW)
        label_output = fn.set_layout(numpy_reader_output1, output_layout=types.NCDHW)
        [roi_start, roi_end] = fn.random_object_bbox(label_output, format="start_end", k_largest=2, foreground_prob=0.4)
        anchor = fn.roi_random_crop(label_output, roi_start=roi_start, roi_end=roi_end, crop_shape=(1, 128, 128, 128))
        data_sliced_output = fn.slice(data_output, anchor=anchor, shape=(1,128,128,128), output_layout=types.NCDHW, output_dtype=types.FLOAT)
        label_sliced_output = fn.slice(label_output, anchor=anchor, shape=(1,128,128,128), output_layout=types.NCDHW, output_dtype=types.UINT8)       
        hflip = fn.random.coin_flip(probability=0.33)
        vflip = fn.random.coin_flip(probability=0.33)
        dflip = fn.random.coin_flip(probability=0.33)
        data_flip_output = fn.flip(data_sliced_output, horizontal=hflip, vertical=vflip, depth=dflip, output_layout=types.NCDHW, output_dtype=types.FLOAT)
        label_flip_output = fn.flip(label_sliced_output, horizontal=hflip, vertical=vflip, depth=dflip, output_layout=types.NCDHW, output_dtype=types.UINT8)
        brightness = fn.random.uniform(range=[0.7, 1.3])
        add_brightness = fn.random.coin_flip(probability=0.1)
        brightness_output = fn.brightness(data_flip_output, brightness=brightness, brightness_shift=0.0, conditional_execution=add_brightness, output_layout=types.NCDHW, output_dtype=types.FLOAT)
        add_noise = fn.random.coin_flip(probability=0.5)
        std_dev = fn.random.uniform(range=[0.0, 0.1])
        noise_output = fn.gaussian_noise(brightness_output, mean=0.0, std_dev=std_dev, conditional_execution=add_noise, output_layout=types.NCDHW, output_dtype=types.FLOAT)
        pipeline.set_outputs(noise_output, label_flip_output)

    pipeline.build()

    pipeline1 = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu, prefetch_queue_depth=6)

    with pipeline1:
        numpy_reader_output = fn.readers.numpy(file_root=data_path, files=x_val, shard_id=local_rank, num_shards=world_size)
        numpy_reader_output1 = fn.readers.numpy(file_root=data_path, files=y_val, shard_id=local_rank, num_shards=world_size)
        data_output = fn.set_layout(numpy_reader_output, output_layout=types.NCDHW)
        label_output = fn.set_layout(numpy_reader_output1, output_layout=types.NCDHW)
        pipeline1.set_outputs(data_output, label_output)

    pipeline1.build()
    
    numpyIteratorPipeline = ROCALNumpyIterator(pipeline, device='cpu' if rocal_cpu else 'gpu')
    print(len(numpyIteratorPipeline))
    valNumpyIteratorPipeline = ROCALNumpyIterator(pipeline1, device='cpu' if rocal_cpu else 'gpu', return_roi=True)
    print(len(valNumpyIteratorPipeline))
    cnt = 0
    for epoch in range(100):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , it in enumerate(numpyIteratorPipeline):
            print(i, it[0].shape, it[1].shape)
            for j in range(batch_size):
                print(it[0][j].cpu().numpy().shape, it[1][j].cpu().numpy().shape)
                cnt += 1
            print("************************************** i *************************************",i)
        numpyIteratorPipeline.reset()
        for i , it in enumerate(valNumpyIteratorPipeline):
            print(i, it[0].shape, it[1].shape)
            for j in range(batch_size):
                print(it[0][j].cpu().numpy().shape, it[1][j].cpu().numpy().shape)
                cnt += 1
            print("************************************** i *************************************",i)
        valNumpyIteratorPipeline.reset()
    print("*********************************************************************")
    print(f'Took {time.time() - start} seconds')

if __name__ == '__main__':
    main()
