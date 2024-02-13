.. meta::
  :description: rocAL documentation and API reference library
  :keywords: rocAL, ROCm, API, documentation

.. _rocal:

********************************************************************
Installation
********************************************************************

This chapter provides information about the installation of rocAL and related packages.  

Prerequisites
=============================

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

Build and install instructions
================================

* [ROCm supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* Install ROCm with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) with `--usecase=graphics,rocm --no-32`

Package install
-------------------------------

Install rocAL runtime, development, and test packages. 
* Runtime package - `rocal` only provides the dynamic libraries
* Development package - `rocal-dev`/`rocal-devel` provides the libraries, executables, header files, and samples
* Test package - `rocal-test` provides ctest to verify installation

On `Ubuntu`
^^^^^^^^^^^^^^^

  ```shell
  sudo apt-get install rocal rocal-dev rocal-test
  ```
On `CentOS`/`RedHat`
^^^^^^^^^^^^^^^^^^^^^

  ```shell
  sudo yum install rocal rocal-devel rocal-test
  ```
On `SLES`
^^^^^^^^^^^^^^

  ```shell
  sudo zypper install rocal rocal-devel rocal-test
  ```

.. note::
    * Package install requires `Turbo JPEG`, `PyBind 11 v2.10.4` and `Protobuf V3.12.4` manual install
    * `CentOS`/`RedHat`/`SLES` requires `FFMPEG Dev` package manual install

Source build and install
==============================

Prerequisites setup script for Linux - rocAL-setup.py
-------------------------------------------------------

For the convenience of the developer, we here provide the setup script which will install all the dependencies required by this project.

.. note::
    This script only needs to be executed once.

Prerequisites for running the script
---------------------------------------

* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`
  + RedHat - `8` / `9`
  + SLES - `15-SP4`
* [ROCm supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* Install ROCm with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) with `--usecase=graphics,rocm --no-32`

**usage:**

```shell
python rocAL-setup.py       --directory [setup directory - optional (default:~/)]
                            --opencv    [OpenCV Version - optional (default:4.6.0)]
                            --protobuf  [ProtoBuf Version - optional (default:3.12.4)]
                            --pybind11  [PyBind11 Version - optional (default:v2.10.4)]
                            --reinstall [Remove previous setup and reinstall (default:OFF)[options:ON/OFF]]
                            --backend   [rocAL Dependency Backend - optional (default:HIP) [options:OCL/HIP]]
                            --rocm_path [ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required]
```
.. note::
    * **ROCm upgrade** requires the setup script **rerun**.

Using rocAL-setup.py
-------------------------
  
* Clone rocAL source code

```shell
git clone https://github.com/ROCm/rocAL.git
```
.. note::
    rocAL has support for two GPU backends: **OPENCL** and **HIP**:

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

.. note::
    + `PyPackageInstall` used for rocal_pybind installation
    + `sudo` required for pybind installation
  
* Instructions for building rocAL with [**OPENCL** GPU backend](https://github.com/ROCm/rocAL/wiki/OpenCL-Backend)

.. note::
    + rocAL_pybind is not supported on OPENCL backend
    + rocAL cannot be installed for both GPU backends in the same default folder (i.e., /opt/rocm/)
    + if an app interested in installing rocAL with both GPU backends, then add **-DCMAKE_INSTALL_PREFIX** in the cmake commands to install rocAL with OPENCL and HIP backends into two separate custom folders.

Verify installation
=========================

* The installer will copy
  + Executables into `/opt/rocm/bin`
  + Libraries into `/opt/rocm/lib`
  + Header files into `/opt/rocm/include/rocal`
  + Apps, & Samples folder into `/opt/rocm/share/rocal`
  + Documents folder into `/opt/rocm/share/doc/rocal`

Verify with rocal-test package
--------------------------------

Test package will install ctest module to test rocAL. Follow below steps to test packge install

```shell
mkdir rocAL-test && cd rocAL-test
cmake /opt/rocm/share/rocal/test/
ctest -VV
```
