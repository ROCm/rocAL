[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="70%" src="https://raw.githubusercontent.com/ROCm/rocAL/master/docs/data/rocAL_logo.png" /></p>

> [!NOTE]
> The published documentation is available at [rocAL](https://rocm.docs.amd.com/projects/rocAL/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

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

### Operating Systems

* Linux distribution
  + Ubuntu - `22.04` / `24.04`
  + RedHat - `8` / `9`
  + SLES - `15 SP7`

### Hardware

* **CPU**: [AMD64](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* **GPU**: [AMD Radeon&trade; Graphics](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) / [AMD Instinct&trade; Accelerators](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

> [!IMPORTANT] 
> * [ROCm-supported hardware required for HIP backend](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
> * `gfx908` or higher GPU required

### Compiler

* AMD Clang++ Version 18.0.0 or later - installed with ROCm

### Libraries
See [installation instructions](#installation-instructions) for more details and instructions on the prerequisite libraries.

#### Required
* MIVisionX (note different installation instructions for package and source)
* CMake (Version `3.10` or later)
* Google Protobuf (Version `3.12.4` or later)
* TurboJPEG (Version `2.0` or later)
* Python3 and Python3 PIP
* Python3 Wheel
* pkg-config
* PyBind11
* RapidJSON

Additional required libraries for source install only:
* rocDecode test package

Additional required libraries for package install only (for source install, these are provided by ROCm `7.13` or later):
* HIP
* Half-precision floating-point library (Version `1.12.0` or higher)

#### Optional (feature-enabling)
These libraries are not required to build rocAL. If a library below isn't found at build/configure time, the corresponding feature is simply disabled and the rest of rocAL builds and works normally.

* LMDB Library (Optional, needed only for Caffe/Caffe2 LMDB reader support)
* FFMPEG (Optional, needed only for software video decode support)
* rocDecode (Optional, needed only for hardware-accelerated video decode; falls back to software decode via FFMPEG if absent)
* rocJPEG (Optional, needed only for hardware-accelerated JPEG decode; falls back to TurboJPEG if absent)
* libsndfile (Optional, needed only for audio pipeline support; also requires MIVisionX `vx_rpp` version `3.1.0` or later)
* Libtar (Optional, needed only for WebDataset reader support)
* DLPack (Optional, pybind-only, enables zero-copy tensor interop with TensorFlow/JAX/generic frameworks)
* hipFile (Optional, HIP backend only, enables GPU Direct Storage I/O for the numpy reader; even when installed, it is disabled at runtime unless `ROCAL_USE_HIPFILE=1` is set)

> [!NOTE]
> * Building the Python bindings (`rocAL_pybind`) links the system Python3 runtime library directly into the core `librocal.so`, not just into the separate pybind module. If you only need the C++ API, be aware that `librocal.so` will still depend on `libpython3.x` whenever Python3/PyBind11 are found at build time.

> [!IMPORTANT]
> * Required compiler support
>   * C++17
>   * OpenMP
>   * Threads
> * On Ubuntu 22.04 - Additional package required: libstdc++-12-dev
>  ```shell
>  sudo apt install libstdc++-12-dev
>  `````

## Installation instructions

> [!IMPORTANT]
> First, install ROCm on your system. Second, install the prerequisites (required for both installation methods).
> Then, choose your installation method based on your environment:
> 1. **ROCm `7.2.x` or below** — install the prebuilt packages (see [package install](#package-install)).
> 2. **ROCm `7.13` or later** — build from source on top of the ROCm Core SDK (see [source install](#source-install)).

### Install the ROCm Core SDK
Follow the [ROCm install guide](https://rocm.docs.amd.com/en/latest/install/rocm.html) for your GPU and operating system. Verify your hardware is on the [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) before proceeding.

### Install prerequisites (required for both package and source installs)
>[!NOTE]
> * All package installs are shown with the `apt` package manager. Use the appropriate package manager for your operating system.

* CMake Version `3.10` or later
  ```shell
  sudo apt install cmake
  ```
* pkg-config
  ```shell
  sudo apt install pkg-config
  ```
* [Google Protobuf](https://developers.google.com/protocol-buffers) - Version `3.12.4` or higher
  ```shell
  sudo apt install libprotobuf-dev protobuf-compiler
  ```

* [TurboJPEG](https://libjpeg-turbo.org/) - Version `2.0` or higher
  ```shell
  sudo apt install libturbojpeg0-dev libjpeg-dev
  ```
>[!NOTE]
  > If TurboJPEG `>= 2.0` is not available to install via your distribution's repository, it must be manually installed. Source: [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo).

* Python3 and Python3 PIP
  ```shell
  sudo apt install python3-dev python3-pip
  ```
* Python3 Wheel
  ```shell
  sudo apt install python3-wheel
  ```

Required manual installs:
* [PyBind11](https://github.com/pybind/pybind11) - Manual install
  * Source: `https://github.com/pybind/pybind11`
  * Tag: [v2.11.1](https://github.com/pybind/pybind11/releases/tag/v2.11.1)

* [RapidJSON](https://github.com/Tencent/rapidjson) - Manual install
  * Source: `https://github.com/Tencent/rapidjson.git`
  * Tag: `master`

### Install optional prerequisites (feature-enabling)
None of the libraries below are required to build rocAL. Install only the ones needed for the optional features you want to use; if a library isn't found at configure time, CMake simply disables the corresponding feature and the rest of the build proceeds normally.

* [LMDB Library](http://www.lmdb.tech/doc/) - needed only for Caffe/Caffe2 LMDB reader support
  ```shell
  sudo apt install liblmdb-dev
  ```

* [FFMPEG](https://www.ffmpeg.org) - needed only for software video decode support
  ```shell
  sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
  ```

* [libsndfile](https://github.com/libsndfile/libsndfile) - Version [1.0.31](https://github.com/libsndfile/libsndfile/releases/tag/1.0.31) or later - needed only for audio pipeline support (also requires MIVisionX `vx_rpp` version `3.1.0` or later)
  ```shell
  sudo apt install libsndfile1-dev
  ```

* [Libtar](https://repo.or.cz/libtar.git) - needed only for WebDataset reader support
  > [!NOTE]
  > Libtar has no standard Ubuntu/distribution package and must be built from source.
  ```shell
  git clone -b v1.2.20 https://repo.or.cz/libtar.git && cd libtar
  autoreconf --force --install && CFLAGS="-fPIC" ./configure
  make -j$(nproc) && sudo make install
  ```

* [DLPack](https://github.com/dmlc/dlpack) - pybind-only, enables zero-copy tensor interop with TensorFlow/JAX/generic frameworks
  ```shell
  sudo apt install libdlpack-dev
  ```

* hipFile - HIP backend only, enables GPU Direct Storage I/O for the numpy reader (provided by ROCm; even when installed, this feature stays off at runtime unless `ROCAL_USE_HIPFILE=1` is set)

### Package install
Available for **ROCm `7.2.x` and below**.

#### Install the additional required prerequisite libraries:
* HIP
  ```shell
  sudo apt install hip-dev
  ```

* [MIVisionX](https://github.com/ROCm/MIVisionX) Components: [AMD OpenVX&trade;](https://github.com/ROCm/MIVisionX/tree/master/amd_openvx) and AMD OpenVX&trade; Extensions: `VX_RPP` and `AMD Media`
  ```shell
  sudo apt install mivisionx-dev
  ```

* [Half-precision floating-point](https://half.sourceforge.net) library - Version `1.12.0` or higher
  ```shell
  sudo apt install half
  ```

#### Install the additional optional prerequisite libraries (hardware-accelerated decode):
* rocDecode - needed only for hardware-accelerated video decode; falls back to FFMPEG software decode if absent
  ```shell
  sudo apt install rocdecode-dev
  ```

* [rocJPEG](https://github.com/ROCm/rocJPEG) - needed only for hardware-accelerated JPEG decode; falls back to TurboJPEG if absent
  ```shell
  sudo apt install rocjpeg-dev
  ```

#### Install rocAL runtime, development, and test packages

* Runtime package - `rocal` only provides the dynamic libraries
* Development package - `rocal-dev`/`rocal-devel` provides the libraries, executables, header files, and samples
* Test package - `rocal-test` provides ctest to verify installation

#### `Ubuntu`

  ```shell
  sudo apt-get install rocal rocal-dev rocal-test
  ```

#### `CentOS`/`RedHat`

  ```shell
  sudo yum install rocal rocal-devel rocal-test
  ```

#### `SLES`

  ```shell
  sudo zypper install rocal rocal-devel rocal-test
  ```

>[!IMPORTANT]
> * `SLES` package install requires `TurboJPEG` manual install
>   ```
>   git clone -b 3.0.2 https://github.com/libjpeg-turbo/libjpeg-turbo.git
>   mkdir tj-build && cd tj-build
>   cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib -DWITH_JPEG8=TRUE ../libjpeg-turbo/
>   make -j8 && sudo make install
>   ```
> * `CentOS`/`RedHat`/`SLES` requires additional `FFMPEG Dev` package manual install
> * rocAL Python module: To use python module, you can set PYTHONPATH:
>   + `export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH`


### Source install
For **ROCm `7.13` and above**.

Install the additional source prerequisites:
* rocDecode test package
  ```shell
  sudo apt install amdrocm-decode-test
  ```
* [MIVisionX](https://github.com/ROCm/MIVisionX) - Manual install
  * Source: `https://github.com/ROCm/MIVisionX`

Then, build rocAL from source and install:

* Clone rocAL source code

```shell
git clone https://github.com/ROCm/rocAL.git
cd rocAL
```

* Run the below commands to build rocAL with the **HIP** GPU backend:
```shell
mkdir build-hip
cd build-hip
cmake ../
make -j8
sudo make install
```
>[!NOTE]
> * `PyPackageInstall` used for rocal_pybind installation


>[!IMPORTANT]
> * Use `-D PYTHON_VERSION_SUGGESTED=3.x` with `cmake` for using a specific Python3 version if required.
> * Use `-D AUDIO_SUPPORT=ON` to enable Audio features, Audio support will be enabled by default with ROCm versions > 6.2

  + run tests - [test option instructions](https://github.com/ROCm/MIVisionX/wiki/CTest)
  ```shell
  make test
  ```

>[!NOTE]
> * Make sure all rocAL required libraries are in your PATH. It may also be necessary to set the LIBVA_DRIVERS_PATH and LD_PRELOAD:
> ```shell
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
> export LIBVA_DRIVERS_PATH=/opt/rocm/lib/rocm_sysdeps/lib
> export LD_PRELOAD=$ROCM_PATH/lib/rocm_sysdeps/lib/librocm_sysdeps_va.so.2:$ROCM_PATH/lib/rocm_sysdeps/lib/librocm_sysdeps_va-drm.so.2
> ```
> * To run tests with verbose option, use `make test ARGS="-VV"`.
## Verify installation

* The installer will copy
  * Executables into `/opt/rocm/bin`
  * Libraries into `/opt/rocm/lib`
  * rocal_pybind into `/opt/rocm/lib`
  * Header files into `/opt/rocm/include/rocal`
  * Apps, & Samples folder into `/opt/rocm/share/rocal`
  * Documents folder into `/opt/rocm/share/doc/rocal`

### Verify with rocal-test package

Test package will install ctest module to test rocAL. Follow below steps to test package install

```shell
mkdir rocAL-test && cd rocAL-test
cmake /opt/rocm/share/rocal/test/
ctest -VV
```
>[!NOTE]
> * Make sure all rocAL required libraries are in your PATH
> * `RHEL`/`SLES` - Export FFMPEG libraries into your PATH 
>     + `export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64/:/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` 
> ```shell
> export PATH=$PATH:/opt/rocm/bin
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
> ```

### Verify rocAL PyBind with rocal-test package

Test package will install ctest module to test rocAL PyBindings. Follow below steps to test package install

```shell
mkdir rocal-pybind-test && cd rocal-pybind-test
cmake /opt/rocm/share/rocal/test/pybind
ctest -VV
```
>[!NOTE]
> * Make sure all rocAL required libraries are in your PATH
> ```shell
> export PATH=$PATH:/opt/rocm/bin
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
> export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
> ```

## Documentation

Run the steps below to build documentation locally.

* Sphinx documentation
```bash
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```
* Doxygen
```bash
doxygen .Doxyfile
```

## Technical support

Please email `mivisionx.support@amd.com` for questions, and feedback on rocAL.

Please submit your feature requests, and bug reports on the [GitHub issues](https://github.com/ROCm/rocAL/issues) page.

## Release notes

### Latest release version

[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/ROCm/rocAL?style=for-the-badge)](https://github.com/ROCm/rocAL/releases)

### Changelog

Review all notable [changes](CHANGELOG.md#changelog) with the latest release

### Tested Configurations

* Linux distribution
  * Ubuntu - `22.04` / `24.04`
  * RedHat - `8` / `9`
  * SLES - `15-SP7`
* ROCm: rocm-core - `7.0.0`+
* MIVisionX - `mivisionx-dev`/`mivisionx-devel`
* rocDecode - `rocdecode-dev`/`rocdecode-devel`
* rocJPEG - `rocjpeg-dev`/`rocjpeg-devel`
* Protobuf - `libprotobuf-dev`/`protobuf-devel`
* TurboJPEG - `libturbojpeg0-dev`/`turbojpeg-devel`
* RapidJSON - `https://github.com/Tencent/rapidjson`
* PyBind11 - [v2.11.1](https://github.com/pybind/pybind11)
* FFMPEG - `ffmpeg` dev package
* libsndfile - [1.0.31](https://github.com/libsndfile/libsndfile/releases/tag/1.0.31)
* Libtar - [v1.2.20](https://repo.or.cz/libtar.git)
* rocAL Setup Script - `V4.1.0`
* Dependencies for all the above packages
