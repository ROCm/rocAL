.. meta::
  :description: rocAL Python APU
  :keywords: rocAL, ROCm, API, Python

********************************************************************
rocAL Python API overview
********************************************************************

The rocAL Python package has been created using Pybind11 which enables data transfer between the rocAL C++ API and Python API.
The ``rocal_pybind`` package includes both PyTorch and TensorFlow framework support and support for multiple data readers such as ``FileReader``, ``COCOReader``, and ``TFRecordReader``.

The rocAL data types are defined in `amd.rocal.types <https://github.com/ROCm/rocAL/blob/develop/rocAL_pybind/amd/rocal/types.py>`_. 

``amd.rocal.fn``
  Contains the image augmentations linked to the rocAL C++ API.

``amd.rocal.decoders``
  Image, video, and audio decoders.

``amd.rocal.readers``
  Image, video, and audio readers.

``amd.rocal.pipeline``
  The pipeline class encapsulates  the data needed to build and run a rocAL graph. This includes support for context and graph creation, functions to verify and run the graph, and data transfer functions.

``amd.rocal.types``
  enums exported from the C++ API to Python.

``amd.rocal.plugin.pytorch``
  PyTorch plugin that includes the ``ROCALGenericIterator`` for Pytorch. The ``ROCALClassificationIterator`` class implements an iterator for image classification that returns labelled images.

``amd.rocal.plugin.tf``
  TensorFlow plugin that includes the ``readers.tfrecord`` TensorFlow reader.

``amd.rocal.plugin.jax``
  JAX plugin that includes the ``ROCALJaxIterator`` for JAX. The ``ROCALJaxIterator`` implements an iterator for running a pipeline over multiple devices.

Mask metadata utilities
=======================

Semantic datasets often include polygon or pixelwise masks alongside bounding boxes. The ``amd.rocal.pipeline.Pipeline`` class exposes helper functions that mirror the rocAL C++ metadata APIs:

* ``Pipeline.get_random_mask_pixel()`` returns one randomly selected pixel coordinate per sample. You can configure the sampling strategy via ``rocalSetRandomPixelMaskConfig`` before building the pipeline.
* ``Pipeline.get_select_mask(mask_ids)`` fetches polygon coordinates for the requested mask ids (for example, ``[0, 1]`` to retrieve the first two polygons in every image).
* ``Pipeline.get_random_object_bbox(format)`` emits bounding boxes for randomly selected objects. The ``format`` argument is an element from ``amd.rocal.types.RocalRandomObjectBBoxFormat`` (``OUT_BOX``, ``OUT_ANCHORSHAPE``, or ``OUT_STARTEND``).

The Python iterator example in :mod:`tests.python_api.rocAL_api_coco_pipeline_segmentation` demonstrates how to read these tensors alongside the decoded images and bounding boxes.


