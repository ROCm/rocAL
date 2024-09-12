<p align="center"><img width="70%" src="docs/data/rocAL_logo.png" /></p>

# Changelog

## Online Documentation

[rocAL Documentation](https://github.com/ROCm/rocAL)

## rocAL 2.1.0 (unreleased)

### Changes
* Setup: rocdecode install disabled
* Package: rocdecode dependency removed

### Removals
* TBA

### Optimizations
* TBA

### Resolved issues
* TBA

### Known issues
* Package install requires `OpenCV` manual install
* CentOS/RedHat/SLES requires `FFMPEG Dev` package manual install
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

## rocAL 2.0.0

### Changes
* Support for audio loader and decoder, which uses libsndfile library to decode wav files
* C++ rocAL audio unit test and python script to run and compare the outputs
* Python support for audio decoders
* Pytorch iterator for Audio
* Python audio unit test, and support to verify outputs
* rocDecode for HW decode
* Support for Audio augmentation - PreEmphasis filter
* Support for reading from file lists in file reader
* Support for Audio augmentation - Spectrogram
* Support for Audio augmentation - ToDecibels
* Support for down-mixing audio channels during decoding
* Support for Audio augmentation - Resample
* Support for TensorTensorAdd and TensorScalarMultiply operations
* Support for Uniform and Normal distribution nodes
* Support for Audio augmentation - NonSilentRegionDetection 
* Support for generic augmentation - Slice 
* Support for generic augmentation - Normalize
* Support for Audio augmentation - MelFilterBank

### Removals
* VX Image processing deprecated

### Optimizations

* Packages - dev & tests
* Tests
* Setup Script
* CentOS 7 support
* SLES 15 SP5 support

### Changed
* ROCm install - use case graphics removed

### Resolved issues
* Tests & readme

### Known issues
* Requires custom dependencies installed

### Tested Configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`
  * RedHat - `8` / `9`
  * SLES - `15-SP5`
* ROCm: rocm-core - `6.2.0.60200`
* RPP - `rpp` & `rpp-dev`/`rpp-devel`
* MIVisionX - `mivisionx` & `mivisionx-dev`/`mivisionx-devel`
* Protobuf - `libprotobuf-dev`/`protobuf-devel`
* RapidJSON - `https://github.com/Tencent/rapidjson`
* Turbo JPEG - [Version 3.0.2](https://libjpeg-turbo.org/)
* PyBind11 - [v2.11.1](https://github.com/pybind/pybind11)
* FFMPEG - `ffmpeg 4` dev package
* OpenCV - `libopencv` / [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* libsndfile - [1.0.31](https://github.com/libsndfile/libsndfile/releases/tag/1.0.31)
* rocAL Setup Script - `V2.6.0`
* Dependencies for all the above packages

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
