# rocAL Audio Unit Tests
This application can be used to verify the functionality of the Audio APIs offered by rocAL.

## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `20.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library (Part of the MIVisionX toolkit)
* Radeon Performance Primitives (RPP)
* MIVisionX
* Sndfile

### Build
  ````
  mkdir build
  cd build
  cmake ../
  make
  ````
### Running the application
  ````
./rocal_audio_unittests <audio-dataset-folder>

Usage: ./rocal_audio_unittests <audio-dataset-folder> <test_case> <downmix> <device-gpu=1/cpu=0>
  ````
