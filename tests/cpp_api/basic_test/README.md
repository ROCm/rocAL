# basic test
This application demonstrates a basic usage of rocAL's C API to load images from the disk and tests the functionality for getting image labels and displays the output images.
<p align="center"><img width="90%" src="https://www.github.com/ROCm/rocAL/docs/data/image_augmentation.png" /></p>

## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `16.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library
* [OpenCV 3.1](https://github.com/opencv/opencv/releases) or higher
* ROCm Performance Primitives (RPP)

### build
  ````
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
  mkdir build
  cd build
  cmake ../
  make
  ````
### running the application
  ```
  ./basic_test <path-to-image-dataset> <label_text_file_path> <test_case:0/1> <processing_device=1/cpu=0>  decode_width decode_height <gray_scale:0/rgb:1> decode_shard_counts
  ```