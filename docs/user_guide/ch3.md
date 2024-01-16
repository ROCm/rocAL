# Chapter 3: Installation

This chapter provides information about the installation of rocAL and related packages.  

## 3.1 Prerequisites 

* Linux distribution
*  [AMD RPP](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp)
*  [AMD OpenVX&trade;](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/amd_openvx) and AMD OpenVX&trade; Extensions: `VX_RPP` and `AMD Media`
*  [Turbo JPEG](https://libjpeg-turbo.org/) - Version `2.0` or higher
*  [Half-precision floating-point](https://half.sourceforge.net) library - Version `1.12.0` or higher
*  [Google Protobuf](https://developers.google.com/protocol-buffers) - Version `3.12.4` or higher
*  [LMBD Library](http://www.lmdb.tech/doc/)
*  [RapidJSON](https://github.com/Tencent/rapidjson)
*  [PyBind11](https://github.com/pybind/pybind11)

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
