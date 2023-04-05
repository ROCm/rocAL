# rocAL Installation

This chapter provides information about the installation of rocAL and related packages.  

## 3.1 Prerequisites 

- Linux Distribution
- [AMD RPP library](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp)
- MIvisionX (AMD RPP OpenVX extension) 
- Boost lib 1.66 or higher
- [Turbo JPEG](https://github.com/rrawther/libjpeg-turbo) with Partial Decoder support
- Half float library
- jsoncpp library
- Google protobuf 3.11.1 or higher
- LMDB

## 3.2 Platform Support

To see the list of supported platforms for rocAL, see the ROCm Installation Guide at https://docs.amd.com. 

## 3.3 Installing rocAL 

rocAL is shipped along with MIVisionX. To build and install the rocAL C++ library, follow the instructions given [here](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX#build--install-mivisionx)

## 3.4 Installing rocAL Python Package 

The rocAL Python package (rocal_pybind) is a separate redistributable wheel. rocal_pybind, which is created using Pybind11, enables data transfer between rocAL C++ API and Python API. With the help of rocal_pybind.so wrapper library, the rocAL functionality, which is primarily in C/C++, can be effectively used in Python. 
The Python package supports PyTorch, TensorFlow, Caffe2, and data readers available for various formats such as FileReader, COCO Reader, TFRecord Reader, and CaffeReader.

To build and install the Python package, see [rocAL python](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/rocAL/rocAL_pybind).

## 3.5 Installing rocAL Using Framework Dockers

To test the rocAL Python APIs using PyTorch or TensorFlow, we recommend building a docker with rocAL and ROCm using any of the links below:

- [rocAL PyTorch docker](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/docker/pytorch)
- [rocAL TensorFlow docker](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/docker/tensorflow)

To use rocAL on Ubuntu, use the following dockers:

- [rocAL on ubuntu20](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/docker/mivisionx-on-ubuntu20.dockerfile)
- [rocAL on Ubuntu22](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/docker/mivisionx-on-ubuntu22.dockerfile)
