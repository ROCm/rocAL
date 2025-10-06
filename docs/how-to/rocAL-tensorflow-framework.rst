.. meta::
  :description: rocAL TensorFlow integration
  :keywords: rocAL, ROCm, API, TensorFlow, training, machine learning, ML, example

*******************************************
Using rocAL with TensorFlow for training
*******************************************

.. _tensorflow:

rocAL improves machine learning (ML) pipeline efficiency by preprocessing data and parallelizing data loading. 

TensorFlow iterators and readers are provided as plugins to separate data loading from training.

You'll need a `rocAL TensorFlow Docker container <https://github.com/ROCm/rocAL/blob/develop/docker/README.md>`_ to run TensorFlow training with rocAL.

To use rocAL with TernsorFlow, import the rocAL TensorFlor plugin:

.. code:: python

  from amd.rocal.plugin.tf import ROCALIterator

Set up a training pipeline that reads data with ``readers.tfrecord`` and uses ``decoders.image`` to decode the raw images. 

Call the training pipeline using ``ROCALIterator``.

An example of TensorFlow training using rocAL is available in the `rocAL GitHub repository <https://github.com/ROCm/rocAL/tree/develop/docs/examples/tf/>`_.
