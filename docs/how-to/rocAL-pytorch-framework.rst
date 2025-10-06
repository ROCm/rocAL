.. meta::
  :description: rocAL PyTorch integration
  :keywords: rocAL, ROCm, API, PyTorch, training, machine learning, ML, example

.. _pytorch:

**************************************
Using rocAL with PyTorch for training
**************************************

rocAL improves machine learning (ML) pipeline efficiency by preprocessing data and parallelizing data loading. 

PyTorch iterators and readers are provided as plugins to separate data loading from training.

You'll need a `rocAL PyTorch Docker container <https://github.com/ROCm/rocAL/blob/develop/docker/README.md>`_ to run PyTorch training with rocAL.

To use rocAL with PyTorch, import the rocAL PyTorch plugin:

.. code:: python

  from amd.rocal.plugin.pytorch import ROCALClassificationIterator

Set up a training pipeline that reads data from a dataset using ``readers.file`` and uses ``decoders.image_slice`` to decode the raw images. 

Call the training pipeline using ``ROCALClassificationIterator``.

Two examples of PyTorch training using rocAL are available in the `rocAL GitHub repository <https://github.com/ROCm/rocAL/blob/master/docs/examples/pytorch/>`_.

