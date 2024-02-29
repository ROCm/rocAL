# rocal_multiple_loaders application
This application demonstrates a basic usage of rocAL's C API to read image files from disk and decode them using different loaders and apply different augmentation in each pipeline.

## Build Instructions

### Pre-requisites
*  Ubuntu 20.04/22.04 Linux
*  rocAL library (Part of the MIVisionX toolkit)
*  [OpenCV 3.1](https://github.com/opencv/opencv/releases) or higher
*  ROCm Performance Primitives (RPP)

### build
  ````
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
  mkdir build
  cd build
  cmake ../
  make
  ````
### Running the application
  ````
./rocAL_multiple_loaders_test <image-dataset-folder> output_image_name <width> <height> gpu=1/cpu=0 rgb=1/grayscale=0 display_all=0(display_last_only)1(display_all)
  ````
