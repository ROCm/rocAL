.. meta::
  :description: rocAL JAX integration
  :keywords: rocAL, ROCm, API, JAX, training, machine learning, ML, example

.. _jax:

**************************************
Using rocAL with JAX for training
**************************************

rocAL improves machine learning (ML) pipeline efficiency by preprocessing data and parallelizing data loading. 

JAX iterators are provided as plugins to separate data loading from training.

You'll need a `rocAL JAX Docker container <https://github.com/ROCm/rocAL/blob/develop/docker/README.md>`_ to run JAX training with rocAL.

To use rocAL with JAX, import the rocAL JAX plugin:

.. code:: python

    from amd.rocal.plugin.jax import ROCALJaxIterator

Get the number of available devices using ``jax.device_count()``. Set up a training pipeline that partitions the training data and run the pipeline on each device using ``ROCALJaxIterator``.

A `Jupyter Notebook <../examples/notebooks/jax_training_example.html>`_ is available as an example of using JAX with rocAL.