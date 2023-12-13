[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="70%" src="docs/data/rocAL_logo.png" /></p>

The AMD ROCm Augmentation Library (**rocAL**) is designed to efficiently decode and process images and videos from a variety of storage formats and modify them through a processing graph programmable by the user. rocAL currently provides C API.
For more details, go to [rocAL user guide](docs) page.

## Documentation

Run the steps below to build documentation locally.

* Sphinx documentation
```bash
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```
* Doxygen
```
doxygen .Doxyfile
```

## Supported Operations

rocAL can be currently used to perform the following operations either with randomized or fixed parameters:

<table>
  <tr>
    <th>Blend</th>
    <th>Blur (Gaussian 3x3)</th> 
    <th>Brightness</th>
    <th>Color Temperature</th>
  </tr>
  <tr>
    <th>ColorTwist</th>
    <th>Contrast</th>
    <th>Crop</th>
    <th>Crop Mirror Normalization</th>
  </tr>
  <tr>
    <th>CropResize</th>
    <th>Exposure Modification</th> 
    <th>Fisheye Lens</th>
    <th>Flip (Horizontal, Vertical and Both)</th>
  </tr>
  <tr>
    <th>Fog</th>
    <th>Gamma</th> 
    <th>Hue</th>
    <th>Jitter</th>
  </tr>
  <tr>
    <th>Lens Correction</th>
    <th>Pixelization</th> 
    <th>Raindrops</th>
    <th>Random Crop</th>
  </tr>
  <tr>
    <th>Resize</th>
    <th>Resize Crop Mirror</th> 
    <th>Rotation</th>
    <th>Salt And Pepper Noise</th>
  </tr>
  <tr>
    <th>Saturation</th>
    <th>Snowflakes</th> 
    <th>Vignette</th>
    <th>Warp Affine</th>
  </tr>
</table>

## Prerequisites

* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7` / `8`
  + RedHat - `8` / `9`
  + SLES - `15-SP4`
*  [ROCm](https://rocmdocs.amd.com/en/latest/deploy/linux/installer/install.html) with --usecase=graphics,rocm
*  [AMD RPP](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp) - MIVisionX Component
*  [AMD OpenVX&trade;](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/amd_openvx) and AMD OpenVX&trade; Extensions: `VX_RPP` and `AMD Media` - MIVisionX Components
*  [Turbo JPEG](https://libjpeg-turbo.org/) - Version 2.0.6.2 from `https://github.com/rrawther/libjpeg-turbo.git`
*  [Half-precision floating-point](https://half.sourceforge.net) library - Version `1.12.0` or higher
*  [Google Protobuf](https://developers.google.com/protocol-buffers) - Version `3.12.4` or higher
*  [LMBD Library](http://www.lmdb.tech/doc/)
*  [RapidJSON](https://github.com/Tencent/rapidjson)
*  [PyBind11](https://github.com/pybind/pybind11)
*  [HIP](https://github.com/ROCm-Developer-Tools/HIP)
*  OpenMP
*  C++17
## Build instructions

### Prerequisites setup script for Linux - `rocAL-setup.py`

For the convenience of the developer, we here provide the setup script which will install all the dependencies required by this project.

  **NOTE:** This script only needs to be executed once.

### Prerequisites for running the script

* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7` / `8`
  + RedHat - `8` / `9`
  + SLES - `15-SP4`
* [ROCm supported hardware](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)
* Install [ROCm](https://rocmdocs.amd.com/en/latest/deploy/linux/installer/install.html) with --usecase=graphics,rocm

  **usage:**

  ```
  python rocAL-setup.py     --directory [setup directory - optional (default:~/)]
                            --opencv    [OpenCV Version - optional (default:4.6.0)]
                            --protobuf  [ProtoBuf Version - optional (default:3.12.4)]
                            --rpp       [RPP Version - optional (default:1.2.0)]
                            --mivisionx [MIVisionX Version - optional (default:master)]
                            --pybind11  [PyBind11 Version - optional (default:v2.10.4)]
                            --reinstall [Remove previous setup and reinstall (default:no)[options:yes/no]]
                            --backend   [rocAL Dependency Backend - optional (default:HIP) [options:OCL/HIP]]
                            --rocm_path [ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required]
  ```
    **Note:**
    * **ROCm upgrade** requires the setup script **rerun**.

### Using `rocAL-setup.py`

* Install [ROCm](https://rocmdocs.amd.com/en/latest/deploy/linux/installer/install.html) with --usecase=graphics,rocm
  
* Use the below commands to set up and build rocAL
  
  + Clone rocAL source code

  ```
  git clone https://github.com/ROCmSoftwarePlatform/rocAL.git
  cd rocAL
  ```
  **Note:** rocAL supports **CPU** and two **GPU** backends: **OPENCL**/**HIP**:

  + Building rocAL with default **HIP** backend:

    + run the setup script to install all the dependencies required
    ```
    python rocAL-setup.py
    ```

    + run the below commands to build rocAL
    ```
    mkdir build-hip
    cd build-hip
    cmake ../
    make -j8
    sudo cmake --build . --target PyPackageInstall
    sudo make install
    ```
    
    + run tests - [test option instructions](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/wiki/CTest)
    ```
    make test
    ```
    
    **Note:** sudo is required to build rocAL_pybind package (only supported on HIP backend)

  **Note:**
  + rocAL_pybind is not supported on OPENCL backend
  + rocAL cannot be installed for both GPU backends in the same default folder (i.e., /opt/rocm/)
  + if an app interested in installing rocAL with both GPU backends, then add **-DCMAKE_INSTALL_PREFIX** in the cmake
  commands to install rocAL with OPENCL and HIP backends into two separate custom folders.

## Tested Configurations

* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`
  + RedHat - `8` / `9`
  + SLES - `15-SP4`
* ROCm: rocm-core - `5.7.0.50700-6`
* RPP - [1.2.0](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/releases/tag/1.2.0)
* MIVisionX - [master](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX)
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* RapidJSON- [master](https://github.com/Tencent/rapidjson)
* PyBind11 - [v2.10.4](https://github.com/pybind/pybind11)
* CuPy - [v12.2.0](https://github.com/ROCmSoftwarePlatform/cupy/releases/tag/v12.0.0)
* rocAL Setup Script - `V1.0.2`
* Dependencies for all the above packages
