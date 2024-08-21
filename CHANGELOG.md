<p align="center"><img width="70%" src="docs/data/rocAL_logo.png" /></p>

# Changelog

## Online Documentation

[rocAL Documentation](https://github.com/ROCm/rocAL)

## rocAL 2.0.0 (unreleased)

### Added

* Packages - dev & tests
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
* Support for downmixing audio channels during decoding
* Support for Audio augmentation - Resample
* Support for TensorTensorAdd and TensorScalarMultiply operations
* Support for Uniform and Normal distribution nodes
* Support for Audio augmentation - NonSilentRegionDetection 
* Support for generic augmentation - Slice 
* Support for generic augmentation - Normalize
* Support for Audio augmentation - MelFilterBank

### Optimizations

* Tests
* Setup Script
* CentOS 7 support
* SLES 15 SP5 support

### Changed

* Image to tensor updates
* ROCm install - use case graphics removed

### Fixed

* Tests & readme

### Tested Configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`
  * RedHat - `8` / `9`
  * SLES - `15-SP5`
* ROCm: rocm-core - `6.1.0.60100-64`
* RPP - `rpp` & `rpp-dev`/`rpp-devel`
* MIVisionX - `mivisionx` & `mivisionx-dev`/`mivisionx-devel`
* rocDecode - `rocdecode` & `rocdecode-dev`/`rocdecode-devel`
* Protobuf - `libprotobuf-dev`/`protobuf-devel`
* RapidJSON - `https://github.com/Tencent/rapidjson`
* Turbo JPEG - [Version 3.0.2](https://libjpeg-turbo.org/)
* PyBind11 - [v2.11.1](https://github.com/pybind/pybind11)
* FFMPEG - `ffmpeg 4` dev package
* OpenCV - `libopencv` / [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* libsndfile - [1.0.31](https://github.com/libsndfile/libsndfile/releases/tag/1.0.31)
* rocAL Setup Script - `V2.5.0`
* Dependencies for all the above packages

### Known issues

* Requires custom deps install

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
