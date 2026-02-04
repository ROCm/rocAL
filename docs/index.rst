.. meta::
  :description: rocAL documentation and API reference library
  :keywords: rocAL, ROCm, API, documentation

.. _rocal:

********************************************************************
rocAL documentation
********************************************************************

The ROCm Augmentation Library (rocAL) is a Python library that provides a way to customize audio, video, and image pipelines for different datasets and models, improving the throughput and performance of deep learning applications. rocAL is optimized for loading and pre-processing data for deep learning applications, with support for multiple data formats and augmentations. 

The rocAL public repository is located at `https://github.com/ROCm/rocAL <https://github.com/ROCm/rocAL>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

      * :doc:`rocAL prerequisites <./install/rocAL-prerequisites>`
      * :doc:`Installing rocAL with the package installer <./install/rocAL-package-install>`
      * :doc:`Building and installing rocAL from source <./install/rocAL-build-and-install>`

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Conceptual

    * :ref:`overview`

  .. grid-item-card:: How to

    * :doc:`Create and run the rocAL pipeline <./how-to/rocAL-use-pipeline>`
    * :doc:`Run PyTorch training with rocAL <./how-to/rocAL-pytorch-framework>`
    * :doc:`Run TensorFlow training with rocAL <./how-to/rocAL-tensorflow-framework>`
    * :doc:`Run JAX training with rocAL <./how-to/rocAL-jax-framework>`

  .. grid-item-card:: Examples
    
    * :doc:`rocAL framework integration examples <./examples/examples>`

  .. grid-item-card:: Reference

    * :doc:`rocAL pipelines <./reference/rocAL-pipeline>`  
    * :doc:`rocAL RNNT dataloading <./reference/rocAL-and-RNNT>`  
    * :doc:`rocAL Python API overview <../reference/rocAL-python-api>`
    * :doc:`rocAL Python reference <../reference/rocAL-python-api-list>`

    The C++ API for contributors:
    
    * :doc:`rocAL C++ API overview <../reference/rocAL-cpp-api>`
    * :doc:`rocAL C++ reference <../reference/rocAL-cpp-api-list>`


To contribute to the documentation refer to `Contributing to ROCm Docs <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.

