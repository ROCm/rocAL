.. meta::
  :description: rocAL documentation and API reference library
  :keywords: rocAL, ROCm, API, documentation

.. _install:

********************************************************************
Installation
********************************************************************

This chapter provides information about the installation of rocAL and related packages.  

Prerequisites
=============================

* Linux distribution

  - Ubuntu 20.04 or 22.04
  - CentOS 7
  - RedHat 8 or 9
  - SLES 15-SP5

* ROCm-supported hardware

* Install ROCm `6.1.0` or later with amdgpu-install: Required usecase - rocm

* HIP

* RPP

* MIVisionX

* rocDecode

* Half-precision floating-point library - Version `1.12.0` or higher

* Google Protobuf

* LMBD Library

* Python3 and Python3 PIP

* Python Wheel

* PyBind11

* Turbo JPEG

* RapidJSON

* **Optional**: FFMPEG

* **Optional**: OpenCV

IMPORTANT
* Compiler features required
  * OpenMP
  * C++17

Installation instructions
================================

The installation process uses the following steps: 

* `ROCm supported hardware install <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_
* Install ROCm with `amdgpu-install <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html>`_ with ``--usecase=rocm``
* Use either :ref:`package-install` or :ref:`source-install` as described below.

.. _package-install:

Package install
-------------------------------

Install rocAL runtime, development, and test packages. 

* Runtime package - ``rocal`` only provides the dynamic libraries
* Development package - ``rocal-dev`` / ``rocal-devel`` provides the libraries, executables, header files, Python bindings, and samples
* Test package - ``rocal-test`` provides ctest to verify installation

On Ubuntu
^^^^^^^^^^^^^^^

.. code-block:: shell

  sudo apt-get install rocal rocal-dev rocal-test


On CentOS/RedHat
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

  sudo yum install rocal rocal-devel rocal-test


On SLES
^^^^^^^^^^^^^^

.. code-block:: shell

  sudo zypper install rocal rocal-devel rocal-test


.. note::
    * Package install requires ``Turbo JPEG``, ``PyBind 11 v2.10.4`` and ``Protobuf V3.12.4`` manual install
    * ``CentOS`` / ``RedHat`` / ``SLES`` requires ``FFMPEG Dev`` package manual install

.. _source-install:

Source Install
---------------------------

For your convenience the ``rocAL-setup.py`` setup script is provided for Linux installations. This script will install all the dependencies required for the rocAL API.

.. note::
    This script only needs to be executed once. However, upgrading the ROCm version also requires rerunning the ``rocAL-setup.py`` script.

The process for installing with the setup script is as follows:

#. Clone rocAL source code

    .. code-block:: shell

      git clone https://github.com/ROCm/rocAL.git

#. Use either flow depending on the backend:

  * :ref:`hip-backend` 
  * :ref:`opencl-backend` 

    .. note::

        rocAL supports two GPU backends: OpenCL and HIP

Running the ``rocAL-setup.py`` setup script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Prerequisites:

  * Linux distribution

    - Ubuntu 20.04 or 22.04
    - CentOS 7
    - RedHat 8 or 9
    - SLES 15-SP5

  * `ROCm supported hardware <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_
  * Install ROCm with `amdgpu-install <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html>`_ with ``--usecase=rocm``

Using ``rocAL-setup.py`` script:

.. code-block:: python

  python rocAL-setup.py       --directory [setup directory - optional (default:~/)]
                              --opencv    [OpenCV Version - optional (default:4.6.0)]
                              --pybind11  [PyBind11 Version - optional (default:v2.10.4)]
                              --reinstall [Remove previous setup and reinstall (default:OFF)[options:ON/OFF]]
                              --backend   [rocAL Dependency Backend - optional (default:HIP) [options:OCL/HIP]]
                              --rocm_path [ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required]


.. _hip-backend:

Instructions for building rocAL with the HIP GPU backend (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Run the setup script to install all the dependencies required by the HIP GPU backend:
  
.. code-block:: shell

  cd rocAL
  python rocAL-setup.py


2. Run the following commands to build rocAL with the HIP GPU backend:
  
.. code-block:: shell

  mkdir build-hip
  cd build-hip
  cmake ../
  make -j8
  sudo cmake --build . --target PyPackageInstall
  sudo make install


3. Run tests - `test option instructions <https://github.com/ROCm/MIVisionX/wiki/CTest>`_
 
.. code-block:: shell

  make test


.. note::
    * `PyPackageInstall` used for rocal_pybind installation
    * `sudo` required for pybind installation
  
.. _opencl-backend:

Instructions for building rocAL with OpenCL GPU backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Find instructions on building rocAL for use with the OpenCL backend on `OPENCL GPU Backend <https://github.com/ROCm/rocAL/wiki/OpenCL-Backend>`_.

.. note::
    * rocAL_pybind is not supported on OPENCL backend
    * rocAL cannot be installed for both GPU backends in the same default folder (i.e., ``/opt/rocm/``)
    * If an app interested in installing rocAL with both GPU backends, then add ``-DCMAKE_INSTALL_PREFIX`` in the cmake commands to install rocAL with OPENCL and HIP backends into two separate custom folders.

Verify installation
=========================

The installer will copy: 

  * Executables into ``/opt/rocm/bin``
  * Libraries into ``/opt/rocm/lib``
  * Header files into ``/opt/rocm/include/rocal``
  * Apps, & Samples folder into ``/opt/rocm/share/rocal``
  * Documents folder into ``/opt/rocm/share/doc/rocal``

Verify with ``rocal-test`` package
--------------------------------------------

Test package will install ctest module to test rocAL. Follow below steps to test package install

.. code-block:: shell

  mkdir rocAL-test && cd rocAL-test
  cmake /opt/rocm/share/rocal/test/
  ctest -VV

