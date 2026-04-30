 .. meta::
  :description: rocAL building and installing
  :keywords: rocAL, ROCm, API, documentation

********************************************************************
Building and installing rocAL from source code
********************************************************************

Before building and installing rocAL, ensure ROCm is installed.

The rocAL source code is available from https://github.com/ROCm/rocAL. The default develop branch is intended for developers who want to contribute to the rocAL project or who want to preview new features.

rocAL supports the HIP backend. 

You can choose to use the |setup|_ setup script to install most :doc:`prerequisites <./rocAL-prerequisites>`

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

.. |setup| replace:: ``rocAL-setup.py``
.. _setup: https://github.com/ROCm/rocAL/blob/develop/rocAL-setup.py
