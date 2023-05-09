[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="70%" src="docs/data/rocAL_logo.png" /></p>

The AMD ROCm Augmentation Library (**rocAL**) is designed to efficiently decode and process images and videos from a variety of storage formats and modify them through a processing graph programmable by the user. rocAL currently provides C API.
For more details, go to [docs](docs) page.

## Documentation

Run the steps below to build documentation locally.

```
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
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
  + SLES - `15-SP2`
*  [AMD RPP](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp)
*  [AMD OpenVX&trade;](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/amd_openvx) and AMD OpenVX&trade; Extensions: `VX_RPP` and `AMD Media`
*  [Boost library](https://www.boost.org) - Version `1.72` or higher
*  [Turbo JPEG](https://libjpeg-turbo.org/) - Version `2.0` or higher
*  [Half-precision floating-point](https://half.sourceforge.net) library - Version `1.12.0` or higher
*  [Google Protobuf](https://developers.google.com/protocol-buffers) - Version `3.12.4` or higher

## Build instructions

### Prerequisites setup script for Linux - `rocAL-setup.py`

For the convenience of the developer, we here provide the setup script which will install all the dependencies required by this project.

  **NOTE:** This script only needs to be executed once.

### Prerequisites for running the script

* Linux distribution
  + Ubuntu - `20.04` / `22.04`
* [ROCm supported hardware](https://docs.amd.com)
* [ROCm](https://docs.amd.com)

  **usage:**

  ```
  python rocAL-setup.py     --directory [setup directory - optional (default:~/)]
                            --opencv    [OpenCV Version - optional (default:4.6.0)]
                            --protobuf  [ProtoBuf Version - optional (default:3.12.4)]
                            --rpp       [RPP Version - optional (default:0.98)]
                            --mivisionx [MIVisionX Version - optional (default:rocm-5.4.1)]
                            --reinstall [Remove previous setup and reinstall (default:no)[options:yes/no]]
                            --backend   [rocAL Dependency Backend - optional (default:HIP) [options:OCL/HIP]]
                            --rocm_path [ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required]
  ```
    **Note:**
    * **ROCm upgrade** with `sudo apt upgrade` requires the setup script **rerun**.
    * use `X Window` / `X11` for [remote GUI app control](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/wiki/X-Window-forwarding)

### Using `rocAL-setup.py`

* Install [ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)
* Use the below commands to set up and build rocAL

  ```
  git clone https://github.com/ROCmSoftwarePlatform/rocAL.git
  cd rocAL
  ```

  **Note:** rocAL has support for two GPU backends: **OPENCL** and **HIP**:

  + Instructions for building rocAL with the **HIP** GPU backend (i.e., default GPU backend):

    + run the setup script to install all the dependencies required by the **HIP** GPU backend:
    ```
    python rocAL-setup.py
    ```

    + run the below commands to build rocAL with the **HIP** GPU backend:
    ```
    mkdir build-hip
    cd build-hip
    sudo cmake ../
    sudo make -j8
    sudo cmake --build . --target PyPackageInstall
    sudo make install
    ```
**Note:** sudo is required to build rocAL_pybind package (only supported on HIP backend)

  + Instructions for building rocAL with **OPENCL** GPU backend:

    + run the setup script to install all the dependencies required by the **OPENCL** GPU backend:
    ```
    python rocAL-setup.py --reinstall yes --backend OCL
    ```

    + run the below commands to build rocAL with the **OPENCL** GPU backend:
    ```
    mkdir build-ocl
    cd build-ocl
    cmake -DBACKEND=OPENCL ../
    make -j8
    sudo make install
    ```

  **Note:**
  + rocAL_pybind is not supported on OPENCL backend
  + rocAL cannot be installed for both GPU backends in the same default folder (i.e., /opt/rocm/)
  if an app interested in installing rocAL with both GPU backends, then add **-DCMAKE_INSTALL_PREFIX** in the cmake
  commands to install rocAL with OPENCL and HIP backends into two separate custom folders.

### Prerequisites - Manual Install

* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* RPP - [0.98](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/releases/tag/0.98)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* MIVisionX - [rocm-5.4.1](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/releases/tag/rocm-5.4.1)
* Turbo JPEG
* LMBD

#### Turbo JPEG installation

Turbo JPEG library is a SIMD optimized library which currently rocAL uses to decode input JPEG images. It needs to be built from the source and installed in the default path for libraries and include headers. You can follow the instruction below to download the source, build and install it.
Note: Make sure you have installed nasm Debian package before installation, it's the dependency required by libturbo-jpeg.

```
 sudo apt-get install nasm
```

Note: You need wget package to download the tar file.

```
 sudo apt-get install wget
```

```
git clone -b 2.0.6.2 https://github.com/rrawther/libjpeg-turbo.git
cd libjpeg-turbo
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_BUILD_TYPE=RELEASE  \
      -DENABLE_STATIC=FALSE       \
      -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.3 \
      -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib  \
      ..
make -j$nproc
sudo make install
```

#### LMDB installation

```
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
```

## Tested Configurations

* Linux distribution
  + Ubuntu - `20.04` / `22.04`
* ROCm: rocm-core - `5.4.0.50400-72`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* RPP - [0.98](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/releases/tag/0.98)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all the above packages
* rocAL Setup Script - `V1.0.0`
