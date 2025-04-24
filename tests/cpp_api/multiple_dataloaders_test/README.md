# multiple_dataloaders_test application
This application demonstrates a basic usage of rocAL's C API to read numpy files from disk and decode them using multiple dataloaders and apply different augmentation in each pipeline.

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
./multiple_dataloaders_test <numpy-dataset-folder> output_image_name gpu=1/cpu=0 display_all=0(display_last_only)1(display_all)
  ````
