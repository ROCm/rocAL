# multiple_dataloaders_test application
This application demonstrates a basic usage of rocAL's C API to read numpy files from disk and decode them using multiple dataloaders and apply different augmentation in each pipeline.

## Build Instructions

### Pre-requisites
* Ubuntu Linux, version `22.04` or later
* rocAL library
* [OpenCV 3.4+](https://github.com/opencv/opencv/releases/tag/3.4.0)
* ROCm Performance Primitives (RPP)

## Build Instructions
````bash
mkdir build
cd build
cmake ../
make
````

## Running the application
```bash
./multiple_dataloaders_test <numpy-dataset-folder> output_image_name gpu=1/cpu=0 display_all=0(display_last_only)1(display_all)
```
