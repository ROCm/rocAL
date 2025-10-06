.. meta::
  :description: rocAL documentation and API reference library
  :keywords: rocAL, ROCm, API, documentation

.. _rocal:

********************************************************************
rocAL documentation
********************************************************************

The ROCm Augmentation Library (rocAL) lets you improve the throughput and performance of your deep learning applications. It's designed to efficiently decode and process image and video pipelines from a variety of storage formats on AMD GPUs and CPUs. Its C++ and Python APIs let you program customizable pipelines for different datasets and models. rocAL is optimized for loading and pre-processing data for deep learning applications, with support for multiple data formats and augmentations. 

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
    * :ref:`architecture`

  .. grid-item-card:: How to

    * :doc:`Run PyTorch training with rocAL <./how-to/rocAL-pytorch-framework>`
    * :doc:`Run TensorFlow training with rocAL <./how-to/rocAL-tensorflow-framework>`
    * :doc:`Run JAX training with rocAL <./how-to/rocAL-jax-framework>`

  .. grid-item-card:: Examples
    
    * `rocAL image processing examples <https://github.com/ROCm/rocAL/tree/master/docs/examples/image_processing>`_ 
    * `rocAL PyTorch examples <https://github.com/ROCm/rocAL/tree/master/docs/examples/pytorch>`_ 
    * `rocAL TensorFlow examples <https://github.com/ROCm/rocAL/tree/master/docs/examples/tf>`_
    * `rocAL Jupyter notebooks <https://github.com/ROCm/rocAL/tree/master/docs/examples/notebooks>`_ 

  .. grid-item-card:: Reference

    * :doc:`rocAL RNNT dataloading <./reference/rocAL-and-RNNT>`  
    * :doc:`rocAL C++ API overview <../reference/rocAL-cpp-api>`
    * :doc:`rocAL C++ reference <../reference/rocAL-cpp-api-list>`
    * :doc:`rocAL Python API overview <../reference/rocAL-python-api>`
    * :doc:`rocAL Python reference <../reference/rocAL-python-api-list>`


To contribute to the documentation refer to `Contributing to ROCm Docs <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.

