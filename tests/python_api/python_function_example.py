
from amd.rocal.plugin.generic import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import os
import cv2
import numpy as np

def generate_random_numbers(count):
    """Generate a list of random numbers."""
    return [1,2,3,4,5]

def generate_random_numbers1(count):
    """Generate a list of random numbers."""
    return [9,9,9,9,9]

def print_output_shape(output):
    print(output.shape)
    return output

def draw_patches(img, idx, device):
    # image is expected as a tensor, bboxes as numpy
    img = img.astype(np.uint8)  # Convert to 8-bit unsigned integers
    # img = img.transpose([0, 2, 3, 1])
    images_list = []
    print("images_list",images_list)
    for im in img:
        images_list.append(im)
    print("images_list",images_list)
    img = cv2.vconcat(images_list)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("eso_blur_host" + str(idx) + ".png", img,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main():
    # Create Pipeline instance
    batch_size = 5
    num_threads = 1
    device_id = 0
    local_rank = 0
    world_size = 1
    rocal_cpu = True
    random_seed = 0
    max_height = 720
    max_width = 640
    color_format = types.RGB
    data_path="/data/MIVisionX-data/rocal_data/coco/coco_10_img_keypoints/person_keypoints_10images_val2017/"
    decoder_device = 'cpu'
    # Execute the pythonScript containing read_array_from_file definition
    data_type = types.FLOAT
    file_path = os.path.abspath(__file__)
    # pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu, tensor_layout=types.NHWC , tensor_dtype=types.INT32, output_memory_type=types.HOST_MEMORY if rocal_cpu else types.DEVICE_MEMORY)
    pipe = Pipeline(batch_size=batch_size, num_threads=8, device_id=device_id,
                                                   seed=random_seed, rocal_cpu=rocal_cpu, tensor_layout=types.NHWC, tensor_dtype=types.FLOAT16)
    with pipe:
        jpegs, _ = fn.readers.file(file_root=data_path)
        images = fn.decoders.image(jpegs,
                                    file_root=data_path,
                                    device=decoder_device,
                                    max_decoded_width=max_width,
                                    max_decoded_height=max_height,
                                    output_type=color_format,
                                    shard_id=local_rank,
                                    num_shards=world_size,
                                    random_shuffle=False)
        output = fn.python_function(images, function = print_output_shape, dtype=types.UINT8, layout=types.NHWC)
        pipe.set_outputs(output)
    pipe.build()
    
    # Dataloader
    data_loader = ROCALClassificationIterator(
        pipe, device="cpu", device_id=local_rank)
    cnt = 0

    # Enumerate over the Dataloader
    for epoch in range(int(1)):
        print("EPOCH:::::", epoch)
        import threading, sys
        print("Main thread owns GIL:", threading.current_thread() is threading.main_thread())
        for i, (output_list, labels) in enumerate(data_loader, 0):
            for j in range(len(output_list)):
                # print("**************", i, "*******************")
                # print("**************starts*******************")
                # print("\nImages:\n", output_list[j])
                # print("\nLABELS:\n", labels)
                # print("**************ends*******************")
                # print("**************", i, "*******************")
                # draw_patches(output_list[j], cnt, "cpu")
                cnt += len(output_list[j])

        data_loader.reset()

if __name__ == '__main__':
    main()
