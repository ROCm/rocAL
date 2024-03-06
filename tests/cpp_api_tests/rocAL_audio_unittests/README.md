# rocAL Unit Tests
This application can be used to verify the functionality of the API offered by rocAL.

## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `16.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library (Part of the MIVisionX toolkit)
* [OpenCV 3.4+](https://github.com/opencv/opencv/releases/tag/3.4.0)
* Radeon Performance Primitives (RPP)

### Build
  ````
  mkdir build
  cd build
  cmake ../
  make
  ````
### Running the application
  ````
./rocAL_audio_unittests

Usage: ./rocAL_audio_unittests <audio-dataset-folder> <test_case> <sample-rate> <downmix> <max_frames> <max_channels> gpu=1/cpu=0
  ````
