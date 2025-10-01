.. meta::
  :description: rocAL Python APU
  :keywords: rocAL, ROCm, API, Python

********************************************************************
rocAL Python API overview
********************************************************************

The rocAL Python package has been created using Pybind11 which enables data transfer between the rocAL C++ API and Python API.
The ``rocal_pybind`` package includes both PyTorch and TensorFlow framework support and support for multiple data readers such as ``FileReader``, ``COCOReader``, and ``TFRecordReader``.

``amd.rocal.fn``
  Contains the image augmentations and file read and decode operations linked to the rocAL C++ API.

``amd.rocal.pipeline``
  The pipeline class encapsulates  the data needed to build and run a rocAL graph. This includes support for context and graph creation, functions to verify and run the graph, and data transfer functions.

``amd.rocal.types``
  enums exported from the C++ API to Python.

``amd.rocal.plugin.pytorch``
  A PyTorch plugin that includes the ``ROCALGenericIterator`` for Pytorch. The ``ROCALClassificationIterator`` class implements an iterator for image classification that returnslabelled images.



