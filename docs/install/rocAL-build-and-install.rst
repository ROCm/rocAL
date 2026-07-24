 .. meta::
  :description: rocAL building and installing
  :keywords: rocAL, ROCm, API, documentation

********************************************************************
Building and installing rocAL from source code
********************************************************************

Before building and installing rocAL, ensure ROCm 7.13 or above is installed, and that all the :doc:`prerequisites <./rocAL-prerequisites>` are installed too. 

The rocAL source code is available from https://github.com/ROCm/rocAL. The default develop branch is intended for developers who want to contribute to the rocAL project or who want to preview new features.

rocAL supports the `HIP backend <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_. 

.. note::
  
  | TurboJPEG must be installed manually on SLES. 
  | To use FFmpeg on SLES and RedHat, the ``FFmpeg-dev`` package must be installed manually.

To build and install rocAL, create the ``build`` directory under the ``rocAL`` root directory. Change directory to ``build``:

.. code:: shell
 
    mkdir build
    cd build

Use ``cmake`` to generate a makefile. Use the ``-DCMAKE_INSTALL_PREFIX`` directive to set the installation directory. For example:

.. code:: shell

    cmake -DCMAKE_INSTALL_PREFIX=/opt/rocAL/


Run make to build and install:

.. code:: shell

  make -j8
  sudo make install
