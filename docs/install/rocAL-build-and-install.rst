 .. meta::
  :description: rocAL building and installing
  :keywords: rocAL, ROCm, API, documentation

********************************************************************
Building and installing rocAL from source code
********************************************************************

Before building and installing rocAL, ensure ROCm has been installed with the `AMDGPU installer <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.1/install/install-methods/amdgpu-installer-index.html>`_ and the ``rocm`` usecase.

The rocAL source code is available from `https://github.com/ROCm/rocAL <https://github.com/ROCm/rocAL>`_. Use the rocAL version that corresponds to the installed version of ROCm.

rocAL supports the HIP backend. 

rocAL is installed in the ROCm installation directory by default. If rocAL for both HIP and OpenCL backends will be installed on the system, each version must be installed in its own custom directory and not in the default directory. 


You can choose to use the |setup| setup script to install most :doc:`prerequisites <./rocAL-prerequisites>`

.. note::
  
  | TurboJPEG must be installed manually on SLES. 
  | To use FFmpeg on SLES and RedHat, the ``FFmpeg-dev`` package must be installed manually.

To build and install rocAL, create the ``build`` directory under the ``rocAL`` root directory. Change directory to ``build``:

.. code:: shell
 
    mkdir build
    cd build

Use ``cmake`` to generate a makefile: 

.. code:: shell
  
  cmake ../

Use the ``-DCMAKE_INSTALL_PREFIX`` directive to set the installation directory. For example:

.. code:: shell

    cmake -DCMAKE_INSTALL_PREFIX=/opt/rocAL/


Run make:

.. code:: shell

  make 

Run ``cmake`` again to generate Python bindings for ``rocal_pybind`` then install:

.. code:: shell

  sudo cmake --build . --target PyPackageInstall
  sudo make install


After the installation, the rocAL files will be installed under ``/opt/rocm/`` unless ``-DCMAKE_INSTALL_PREFIX`` was specified. If ``-DCMAKE_INSTALL_PREFIX`` was specified, the rocAL files will be installed under the specified directory.

To make and run the tests, use ``make test``.

.. |setup| replace:: ``rocAL-setup.py``
.. _openvx: https://github.com/ROCm/rocAL/blob/develop/rocAL-setup.py
