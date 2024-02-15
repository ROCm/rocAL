.. meta::
  :description: rocAL documentation and API reference library
  :keywords: rocAL, ROCm, API, documentation

.. _rocal:

********************************************************************
rocAL documentation
********************************************************************

Deep learning applications require loading and pre-processing data efficiently to achieve high processing throughput. 
This requires creating efficient processing pipelines fully utilizing the underlying hardware capabilities. 
Some examples are load and decode data, and perform a variety of augmentations such as color-format conversions. Deep learning 
frameworks require supporting multiple data formats and augmentations to adapt to a variety of data-sets and models.

The ROCm Augmentation Library (rocAL) is designed to efficiently decode and process image and video pipelines from a 
variety of storage formats. These pipelines are programmable by the user using both C++ and Python APIs. rocAL is 
implemented in the `HIP programming language <https://rocm.docs.amd.com/projects/HIP/>`_ and optimized for AMD's
latest discrete GPUs.

The code is open and hosted at: https://github.com/ROCm/rocAL

The rocAL documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :ref:`install`

  .. grid-item-card:: How-to

    * :ref:`overview`
    * :ref:`architecture`
    * :ref:`using-with-cpp`
    * :ref:`using-with-python`
    * :ref:`framework`

  .. grid-item-card:: Reference

    * `rocAL API Modules <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/modules.html>`_
    * `rocAL API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal.html>`_
    * `rocAL Datatypes <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__types.html>`_
    * `rocAL Augmentation API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__augmentations.html>`_
    * `rocAL Data Loaders API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__data__loaders.html>`_
    * `rocAL Data Transfer API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__data__transfer.html>`_
    * `rocAL Info API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__info.html>`_
    * `rocAL Metadata API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__meta__data.html>`_
    * `rocAL Parameter API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__parameters.html>`_
    * `rocAL Header Files <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/files.html>`_

  .. grid-item-card:: Tutorials

    * :ref:`examples`

To contribute to the documentation refer to `Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/index.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.

