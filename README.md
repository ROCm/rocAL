[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="70%" src="https://raw.githubusercontent.com/ROCm/rocAL/master/docs/data/rocAL_logo.png" /></p>

The AMD ROCm Augmentation Library (**rocAL**) is designed to efficiently decode and process images and videos from a variety of storage formats and modify them through a processing graph programmable by the user. rocAL currently provides C API.
For more details, go to [rocAL user guide](docs) page.

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
  + CentOS - `7`
  + RedHat - `8` / `9`
  + SLES - `15-SP4`
* [ROCm supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* Install ROCm with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) with `--usecase=graphics,rocm --no-32`
*  [RPP](https://github.com/ROCm/rpp)
*  [AMD OpenVX&trade;](https://github.com/ROCm/MIVisionX/tree/master/amd_openvx) and AMD OpenVX&trade; Extensions: `VX_RPP` and `AMD Media` - MIVisionX Components
*  [Turbo JPEG](https://libjpeg-turbo.org/) - Version 2.0.6.2 from `https://github.com/rrawther/libjpeg-turbo.git`
*  [Half-precision floating-point](https://half.sourceforge.net) library - Version `1.12.0` or higher
*  [Google Protobuf](https://developers.google.com/protocol-buffers) - Version `3.12.4` or higher
*  [LMBD Library](http://www.lmdb.tech/doc/)
*  [RapidJSON](https://github.com/Tencent/rapidjson)
*  [PyBind11](https://github.com/pybind/pybind11)
*  [HIP](https://github.com/ROCm/HIP)
*  OpenMP
*  C++17

## Build and install instructions

* [ROCm supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* Install ROCm with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) with `--usecase=graphics,rocm --no-32`

### Package install

Install rocAL runtime, development, and test packages. 
* Runtime package - `rocal` only provides the dynamic libraries
* Development package - `rocal-dev`/`rocal-devel` provides the libraries, executables, header files, and samples
* Test package - `rocal-test` provides ctest to verify installation

##### On `Ubuntu`
  ```shell
  sudo apt-get install rocal rocal-dev rocal-test
  ```
##### On `CentOS`/`RedHat`
  ```shell
  sudo yum install rocal rocal-devel rocal-test
  ```
##### On `SLES`
  ```shell
  sudo zypper install rocal rocal-devel rocal-test
  ```

  **Note:**
  * Package install requires `Turbo JPEG`, `PyBind 11 v2.10.4` and `Protobuf V3.12.4`  manual install
  * `CentOS`/`RedHat`/`SLES` requires `FFMPEG Dev` package manual install

#### Source build and install

### Prerequisites setup script for Linux - rocAL-setup.py

For the convenience of the developer, we here provide the setup script which will install all the dependencies required by this project.

**NOTE:** This script only needs to be executed once.

### Prerequisites for running the script

* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`
  + RedHat - `8` / `9`
  + SLES - `15-SP4`
* [ROCm supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* Install ROCm with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) with `--usecase=graphics,rocm --no-32`

**usage:**

```
python rocAL-setup.py       --directory [setup directory - optional (default:~/)]
                            --opencv    [OpenCV Version - optional (default:4.6.0)]
                            --protobuf  [ProtoBuf Version - optional (default:3.12.4)]
                            --pybind11  [PyBind11 Version - optional (default:v2.10.4)]
                            --reinstall [Remove previous setup and reinstall (default:OFF)[options:ON/OFF]]
                            --backend   [rocAL Dependency Backend - optional (default:HIP) [options:OCL/HIP]]
                            --rocm_path [ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required]
```
**Note:**
  * **ROCm upgrade** requires the setup script **rerun**.

### Using rocAL-setup.py
  
* Clone rocAL source code

```
git clone https://github.com/ROCm/rocAL.git
```
  **Note:** rocAL has support for two GPU backends: **OPENCL** and **HIP**:

* Instructions for building rocAL with the **HIP** GPU backend (default GPU backend):

    + run the setup script to install all the dependencies required by the **HIP** GPU backend:
  
  ```shell
  cd rocAL
  python rocAL-setup.py
  ```

    + run the below commands to build rocAL with the **HIP** GPU backend:

  ```shell
  mkdir build-hip
  cd build-hip
  cmake ../
  make -j8
  sudo cmake --build . --target PyPackageInstall
  sudo make install
  ```

    + run tests - [test option instructions](https://github.com/ROCm/MIVisionX/wiki/CTest)

  ```shell
  make test
  ```

    **Note:**
    + `PyPackageInstall` used for rocal_pybind installation
    + `sudo` required for pybind installation
  
* Instructions for building rocAL with [**OPENCL** GPU backend](https://github.com/ROCm/rocAL/wiki/OpenCL-Backend)

  **Note:**
  + rocAL_pybind is not supported on OPENCL backend
  + rocAL cannot be installed for both GPU backends in the same default folder (i.e., /opt/rocm/)
  + if an app interested in installing rocAL with both GPU backends, then add **-DCMAKE_INSTALL_PREFIX** in the cmake
  commands to install rocAL with OPENCL and HIP backends into two separate custom folders.

## Verify installation

* The installer will copy
  + Executables into `/opt/rocm/bin`
  + Libraries into `/opt/rocm/lib`
  + Header files into `/opt/rocm/include/rocal`
  + Apps, & Samples folder into `/opt/rocm/share/rocal`
  + Documents folder into `/opt/rocm/share/doc/rocal`

### Verify with rocal-test package

Test package will install ctest module to test rocAL. Follow below steps to test packge install

```shell
mkdir rocAL-test && cd rocAL-test
cmake /opt/rocm/share/rocal/test/
ctest -VV
```

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

## Technical support

Please email `mivisionx.support@amd.com` for questions, and feedback on MIVisionX.

Please submit your feature requests, and bug reports on the [GitHub issues](https://github.com/ROCm/rocAL/issues) page.

## Release notes

### Latest release version

[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/ROCm/rocAL?style=for-the-badge)](https://github.com/ROCm/rocAL/releases)

### Changelog

Review all notable [changes](CHANGELOG.md#changelog) with the latest release

### Tested Configurations

* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`
  + RedHat - `8` / `9`
  + SLES - `15-SP4`
* ROCm: rocm-core - `5.7.0.50700-6`
* RPP - `rpp` & `rpp-dev`/`rpp-devel`
* MIVisionX - `mivisionx` & `mivisionx-dev`/`mivisionx-devel`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* RapidJSON- [master](https://github.com/Tencent/rapidjson)
* PyBind11 - [v2.10.4](https://github.com/pybind/pybind11)
* rocAL Setup Script - `V1.1.0`
* Dependencies for all the above packages
