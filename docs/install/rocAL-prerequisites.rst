.. meta::
  :description: rocAL prerequisites
  :keywords: rocAL, ROCm, API, installation, prerequisites

.. _install:

********************************************************************
rocAL prerequisites
********************************************************************

:doc:`Building rocAL from source <./rocAL-build-and-install>` requires CMake 3.10 or later, ``amdclang++`` 18.0.0 or later, and the following compiler support:

* C++17
* OpenMP
* Threads

The installation method used for rocAL depends on the version of ROCm. For ROCm 7.2.x or below, follow the :doc:`package install method <./rocAL-package-install>`. For ROCm 7.13 or above, :doc:`build rocAL from source <./rocAL-build-and-install>` on top of the ROCm Core SDK.

The following prerequisites are required by both installation methods:

* cmake
* pkg-config 
* `Google Protobuf <https://developers.google.com/protocol-buffers>`_ version 3.12.4 or later
* `TurboJPEG <https://libjpeg-turbo.org/>`_
* Python3, Python3 pip, and Python3 wheel
* `LMDB Library <http://www.lmdb.tech/doc/>`_
* `FFMPEG <https://www.ffmpeg.org>`_
* `PyBind11 <https://github.com/pybind/pybind11/releases/tag/v2.11.1>`_ version 2.11.1 (Manual install from `<https://github.com/pybind/pybind11>`_)
* `RapidJSON <https://github.com/Tencent/rapidjson>`_ (Manual install from `<https://github.com/Tencent/rapidjson.git>`_)

.. note::

  | TurboJPEG must be installed manually on SLES. 
  | To use FFMPeg on SLES and RHEL, the ``FFMPeg-dev`` package must be installed manually.
  | libstdc++-12-dev is required on Ubuntu 22.04 only and must be installed manually.

Use the following commands to install those packages (make sure to also manually install the other prerequisites):

.. tab-set::
 
  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt install cmake
      sudo apt install pkg-config
      sudo apt install libprotobuf-dev protobuf-compiler
      sudo apt install libturbojpeg0-dev libjpeg-dev
      sudo apt install python3-dev python3-pip python3-wheel
      sudo apt install liblmdb-dev # optional: needed for Caffe/Caffe2 LMDB reader support
      sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install cmake
      sudo yum install pkg-config
      sudo yum install protobuf-devel protobuf-compiler
      sudo yum install turbojpeg-devel libjpeg-turbo-devel
      sudo yum install python3-devel python3-pip python3-wheel
      sudo yum install lmdb-devel # optional: needed for Caffe/Caffe2 LMDB reader support

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install cmake
      sudo zypper install pkg-config
      sudo zypper install protobuf-devel libprotobuf-c-devel
      sudo zypper install python3-devel python3-pip python3-wheel
      sudo zypper install lmdb-devel # optional: needed for Caffe/Caffe2 LMDB reader support

For source install, it is also required to install MIVisionX manually (`<https://github.com/ROCm/MIVisionX>`_).

For package install, it is also required to install the following prerequisites:

* `MIVisionX <https://rocm.docs.amd.com/projects/MIVisionX/en/latest/index.html>`_ with |openvx|_ and the VX_RPP and AMD Media extensions
* `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_
* `The half-precision floating-point library <https://half.sourceforge.net>`_ version 1.12.0 or later
* `rocDecode <https://rocm.docs.amd.com/projects/rocDecode/en/latest/index.html>`_ used as the hardware video decoder
* `rocJPEG <https://rocm.docs.amd.com/projects/rocJPEG/en/latest/index.html>`_ used as the hardware image decoder

Use the following commands to install these:

.. tab-set::

  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt install mivisionx-dev
      sudo apt install hip-dev
      sudo apt install half
      sudo apt install rocdecode-dev
      sudo apt install rocjpeg-dev

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install mivisionx-devel
      sudo yum install hip-devel
      sudo yum install half
      sudo yum install rocdecode-devel
      sudo yum install rocjpeg-devel

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install mivisionx-devel
      sudo zypper install hip-devel
      sudo zypper install half
      sudo zypper install rocdecode-devel
      sudo zypper install rocjpeg-devel



.. |trade| raw:: html

    &trade;

.. |openvx| replace:: AMD OpenVX\ |trade|
.. _openvx: https://rocm.docs.amd.com/projects/MIVisionX/en/latest/install/MIVisionX-install-OpenVX.html

