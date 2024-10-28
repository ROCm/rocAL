<p align="center"><img width="70%" src="docs/data/rocAL_logo.png" /></p>

# Changelog

## Online Documentation

[rocAL Documentation](https://github.com/ROCm/rocAL)

## rocAL 2.1.0 for ROCm 6.3.0

### Changes
* Setup: rocdecode install disabled
* Package: rocdecode dependency removed
* rocAL Pybind support for package install - rocAL Python module: To use python module, you can set PYTHONPATH:
  + `export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH`
* Add Last batch policy, pad last batch, stick to shard and shard size support for the following image readers : coco, caffe, caffe2, mxnet, tf and cifar10

### Removals
* rocdecode dependencies for package install removed

### Optimizations
* CTest updates

### Resolved issues
* Test failures fixed

### Known issues
* Package install requires `TurboJPEG`, and `RapidJSON` manual install
* `CentOS`/`RedHat`/`SLES` requires additional `FFMPEG Dev` package manual install
* Hardware decode requires rocm usecase `graphics`
 
### Upcoming changes
* Optimized audio augmentations support

### Tested Configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`
  * RedHat - `8` / `9`
  * SLES - `15-SP5`
* ROCm: rocm-core - `6.3.0.60300`
* RPP - `rpp` & `rpp-dev`/`rpp-devel`
* MIVisionX - `mivisionx` & `mivisionx-dev`/`mivisionx-devel`
* Protobuf - `libprotobuf-dev`/`protobuf-devel`
* RapidJSON - `https://github.com/Tencent/rapidjson`
* Turbo JPEG - [Version 3.0.2](https://libjpeg-turbo.org/)
* PyBind11 - [v2.11.1](https://github.com/pybind/pybind11)
* FFMPEG - `ffmpeg 4` dev package
* OpenCV - `libopencv` / [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* libsndfile - [1.0.31](https://github.com/libsndfile/libsndfile/releases/tag/1.0.31)
* rocAL Setup Script - `V2.5.0`
* Dependencies for all the above packages

## rocAL 2.0.0 for ROCm 6.2.1

### Changes

* The new version of rocAL introduces many new features, but does not modify any of the existing public API functions.However, the version number was incremented from 1.3 to 2.0.
  Applications linked to version 1.3 must be recompiled to link against version 2.0.
* Added development and test packages.
* Added C++ rocAL audio unit test and Python script to run and compare the outputs.
* Added Python support for audio decoders.
* Added Pytorch iterator for audio.
* Added Python audio unit test and support to verify outputs.
* Added rocDecode for HW decode.
* Added support for:
  * Audio loader and decoder, which uses libsndfile library to decode wav files
  * Audio augmentation - PreEmphasis filter, Spectrogram, ToDecibels, Resample, NonSilentRegionDetection. MelFilterBank
  * Generic augmentation - Slice, Normalize
  * Reading from file lists in file reader
  * Downmixing audio channels during decoding
  * TensorTensorAdd and TensorScalarMultiply operations
  * Uniform and Normal distribution nodes
* Image to tensor updates
* ROCm install - use case graphics removed

### Known issues

* Dependencies are not installed with the rocAL package installer. Dependencies must be installed with the prerequisite setup script provided. See the [rocAL README on GitHub](https://github.com/ROCm/rocAL/blob/docs/6.2.1/README.md#prerequisites-setup-script) for details.

### **rocBLAS** (4.2.1)

#### Removals

* Removed Device_Memory_Allocation.pdf link in documentation.

#### Resolved issues

* Fixed error/warning message during `rocblas_set_stream()` call.

## rocAL 1.0.0

### Added

* rocAL Tests

### Optimizations

* Image augmentations

### Changed

* Deps

### Fixed

* minor issues

### Tested Configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
* ROCm: rocm-core - `6.0.60002-1`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* RPP - [1.4.0](https://github.com/ROCms/rpp/releases/tag/1.4.0)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* MIVisionX - [master](https://github.com/ROCm/MIVisionX)
* Dependencies for all the above packages
* rocAL Setup Script - `V1.0.2`

### Known issues

* Requires custom version of libturbo-JPEG
