.. meta::
  :description: rocAL JAX integration
  :keywords: rocAL, ROCm, API, JAX, training, machine learning, ML, example

.. _jax:

**************************************
Using rocAL with JAX for training
**************************************

The Jax plugin for rocAL can process the entire dataset at once or it can divide the dataset into shards that are distributed over a mesh of GPUs.

.. code:: 

  from jax.sharding import PositionalSharding
  from jax.experimental import mesh_utils

  mesh = mesh_utils.create_device_mesh((jax.device_count(), 1))
  sharding = PositionalSharding(mesh)

.. note::

  The |jax.py|_ sample and the :doc:`Jax Jupyter notebook <../examples/notebooks/jax_training_example>` both use the ``PositionalSharding()`` helper function to automatically divide the dataset into shards. In later versions of Jax, this function has been deprecated. The ``NamedSharding()`` function is used instead.

When the dataset is divided over multiple GPUs, one :doc:`pipeline <../reference/rocAL-pipeline>` is assigned to each GPU. The pipelines process the data shard assigned to the GPU. 

The pipelines are then processed by the iterators. Two rocAL Jax iterators are available:

* ``ROCALJaxIterator`` is used for general data processing pipelines.
* ``ROCALPeekableIterator`` is a peekable version of ``ROCALJaxIterator`` that lets you peek at the next item without consuming it.

Both Jax iterators can take a single pipeline if the dataset is being processed all at once, or an array of pipelines and the sharding value if the dataset has been divided over a mesh of GPUs.

Pipelines are created by either instantiating them with ``Pipeline()`` or decorating a graph function with ``@pipeline_def``. 

In the :doc:`Jax Jupyter notebook <../examples/notebooks/jax_training_example>`, training is done over a mesh of GPUs. 

Each training pipeline is instantiated, populated with graph elements, and built, before being added to the array of pipelines:

.. code:: python

  from amd.rocal.pipeline import Pipeline
  from amd.rocal.plugin.jax import ROCALJaxIterator

  [...]

   train_pipelines = []
    for id in range(device_count):
    train_pipeline = Pipeline(batch_size=batch_size, num_threads=8, device_id=id, seed=id+42, rocal_cpu=False, tensor_dtype = types.FLOAT, tensor_layout=types.NCHW, prefetch_queue_depth = 3, mean=[0.5 * 255,0.5 * 255,0.5 * 255], std = [0.5 * 255,0.5 * 255,0.5 * 255], output_memory_type = types.DEVICE_MEMORY)

    with train_pipeline:
        cifar10_reader_output = fn.readers.cifar10(file_root=f'{data_path}/cifar-10-batches-bin', shard_id=id, num_shards=device_count, filename_prefix='data_batch_', random_shuffle=True, last_batch_policy=types.LAST_BATCH_DROP)
        cmnp = fn.crop_mirror_normalize(cifar10_reader_output,
                                            output_layout = types.NCHW,
                                            output_dtype = types.FLOAT,
                                            crop=(32, 32),
                                            mirror=0,
                                            mean=[0.5 * 255,0.5 * 255,0.5 * 255],
                                            std=[0.5 * 255,0.5 * 255,0.5 * 255])
        train_pipeline.set_outputs(cmnp)

    train_pipeline.build()
    train_pipelines.append(train_pipeline)
  
  training_iterator = ROCALJaxIterator(train_pipelines, sharding)


The pipelines are then passed to the iterator:

.. code:: python

    imageIteratorPipeline = ROCALJaxIterator(pipelines, sharding=sharding)

Jax isn't tied to a specific data loader and can use any dataset reader.

The validation pipeline in the :doc:`Jax Jupyter notebook <../examples/notebooks/jax_training_example>` processes the entire dataset without sharding:

.. code:: python

  val_pipeline = Pipeline(batch_size=batch_size, num_threads=8, device_id=0, seed=42, rocal_cpu=False, tensor_dtype = types.FLOAT, tensor_layout=types.NCHW, prefetch_queue_depth = 3, mean=[0.5 * 255,0.5 * 255,0.5 * 255], std = [0.5 * 255,0.5 * 255,0.5 * 255], output_memory_type = types.DEVICE_MEMORY)

  with val_pipeline:
    val_cifar10_reader_output = fn.readers.cifar10(file_root=f'{data_path}/cifar-10-batches-bin', shard_id=0, num_shards=1, filename_prefix='test_batch', last_batch_policy=types.LAST_BATCH_DROP)
    val_cmnp = fn.crop_mirror_normalize(val_cifar10_reader_output,
                                            output_layout = types.NCHW,
                                            output_dtype = types.FLOAT,
                                            crop=(32, 32),
                                            mirror=0,
                                            mean=[0.5 * 255,0.5 * 255,0.5 * 255],
                                            std=[0.5 * 255,0.5 * 255,0.5 * 255])
    val_pipeline.set_outputs(val_cmnp)

  val_pipeline.build()
  validation_iterator = ROCALJaxIterator(val_pipeline)

`Prebuilt Docker images with Jax pre-installed <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html#using-docker-with-jax-pre-installed>`_ are available.


.. |jax.py| replace:: ``jax_classification_reader.py``
.. _jax.py: https://github.com/ROCm/rocAL/tree/develop/tests/python_api/jax_classification_reader.py


.. |pipeline.py| replace:: ``pipeline.py``
.. _pipeline.py: https://github.com/ROCm/rocAL/tree/develop/tests/python_api/pipeline.py

