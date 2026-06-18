# dataloader tensorflow application

This application demonstrates a basic usage of rocAL's C API to load TfRecords from the disk and modify them in different possible ways and displays the output images.

## Pre-requisites

*  Ubuntu 16.04/18.04 Linux
*  Optional: OpenCV for display - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
*  Google protobuf 3.11.1 or higher
*  ROCm Performance Primitives (RPP)

## Build Instructions

  ````shell
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
  mkdir build
  cd build
  cmake ../
  make
  ````

### running the application

  ````shell
  ./dataloader_tf <path-to-TFRecord-folder - required> <proc_dev> <decode_width> <decode_height> <batch_size> <grayscale/rgb> <dispay_on_or_off>
  ````
