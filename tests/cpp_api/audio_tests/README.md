# rocAL Audio Unit Tests
This application can be used to verify the functionality of the Audio APIs offered by rocAL.

## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `20.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library
* Radeon Performance Primitives (RPP)
* MIVisionX
* libsndfile

### Build
  ````
  mkdir build
  cd build
  cmake ../
  make
  ````
### Running the application
  ````
./audio_tests <audio-dataset-folder>

Usage: ./audio_tests <audio-dataset-folder> <test_case> <downmix> <device-gpu=1/cpu=0> <qa_mode>
  ````

### Output verification 

The python script `audio_tests.py` can be used to run all test cases for audio functionality in rocAL and verify the correctness of the generated outputs with the golden outputs.

Input data is available in the following link : [MIVisionX-data](https://github.com/ROCm/MIVisionX-data)

`export ROCAL_DATA_PATH=<absolute_path_to_MIVisionX_data>`

```
python3 audio_tests.py --gpu <0/1> --downmix <True/False> --test_case <case_number> --qa_mode <0/1>
```

**Available Test Cases**
* Case 0 - Audio Decoder
* Case 1 - PreEmphasis Filter
* Case 2 - Spectrogram
* Case 3 - Downmix
* Case 4 - ToDecibels
* Case 5 - Resample
* Case 6 - TensorAddTensor
* Case 7 - TensorMulScalar
* Case 8 - NonSilentRegionDetection
* Case 9 - Slice
* Case 10 - MelFilterBank
* Case 11 - Normalize