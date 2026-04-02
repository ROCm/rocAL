.. meta::
  :description: rocAL PyTorch integration
  :keywords: rocAL, ROCm, API, PyTorch, training, machine learning, ML, example

.. _pytorch:

**************************************
Using rocAL with PyTorch for training
**************************************

The PyTorch plugin for rocAL includes three iterators that can be used to process different types of pipelines.

* ``ROCALAudioIterator`` is used for audio processing pipelines.
* ``ROCALGenericIterator`` is used for general data processing pipelines.
* ``ROCALClassificationIterator`` is used for classification and training pipelines.

The iterators run the training and validation :doc:`pipelines <../reference/rocAL-pipeline>`, prefetching and loading the next batch of files while the previous batch is being processed.

Pipelines are created by either instantiating them with ``Pipeline()`` or decorating a graph function with ``@pipeline_def``. 

The training and validation pipelines in |imagenet.py|_ are both instantiated with ``Pipeline()``:

.. code:: python

  from amd.rocal.pipeline import Pipeline

  [...]

  def train_pipeline(data_path, batch_size, local_rank, world_size, num_thread, crop, rocal_cpu, fp16):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_thread, device_id=local_rank, seed=local_rank+10, rocal_cpu=rocal_cpu, tensor_dtype=types.FLOAT16 if fp16 else types.FLOAT, tensor_layout=types.NCHW,
                    prefetch_queue_depth=6, mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255], output_memory_type=types.HOST_MEMORY if rocal_cpu else types.DEVICE_MEMORY)
    with pipe:
        jpegs, labels = fn.readers.file(file_root=data_path)
        rocal_device = 'cpu' if rocal_cpu else 'gpu'
        decode = fn.decoders.image_slice(jpegs, output_type=types.RGB,
                                         file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        res = fn.resize(decode, resize_width=224, resize_height=224, output_layout=types.NHWC,
                        output_dtype=types.UINT8, interpolation_type=types.TRIANGULAR_INTERPOLATION)
        flip_coin = fn.random.coin_flip(probability=0.5)
        cmnp = fn.crop_mirror_normalize(res,
                                        output_layout=types.NCHW,
                                        output_dtype=types.FLOAT,
                                        crop=(crop, crop),
                                        mirror=flip_coin,
                                        mean=[0.485 * 255, 0.456 *
                                              255, 0.406 * 255],
                                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        pipe.set_outputs(cmnp)
    print('rocal "{0}" variant'.format(rocal_device))
    return pipe

Data is read from the dataset using ``readers.file``, which is appropriate for PyTorch:

.. code::

  import amd.rocal.fn as fn

  [...]

  with pipe:
        jpegs, labels = fn.readers.file(file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
  

The appropriate iterator is then used to load data and run the pipeline. In |imagenet.py|, the ``ROCALClassificationIterator`` is used.

.. code:: python

  from amd.rocal.plugin.pytorch import ROCALClassificationIterator

  [...]

  def get_rocal_train_loader(data_path, batch_size, local_rank, world_size, num_thread, crop, rocal_cpu, fp16=False):
    traindir = os.path.join(data_path, 'train')
    pipe_train = train_pipeline(
        traindir, batch_size, local_rank, world_size, num_thread, crop, rocal_cpu, fp16)
    pipe_train.build()
    train_loader = ROCALClassificationIterator(
        pipe_train, device="cpu" if rocal_cpu else "cuda", device_id=local_rank)
    return Prefetcher(train_loader, rocal_cpu, batch_size)

Two examples of PyTorch training using rocAL are available in the `rocAL GitHub repository <https://github.com/ROCm/rocAL/blob/develop/docs/examples/pytorch/>`_.  `Jupyter Notebooks <https://github.com/ROCm/rocAL/tree/develop/docs/examples/notebooks>`_ are also available.

A `Docker container <https://github.com/ROCm/rocAL/blob/develop/docker/README.md>`_ is available for PyTorch training with rocAL.


.. |imagenet.py| replace:: ``imagenet_training.py``
.. _imagenet.py: https://github.com/ROCm/rocAL/tree/develop/docs/examples/pytorch/imagenet_training/imagenet_training.py

.. |train.py| replace:: ``train.py``
.. _train.py: https://github.com/ROCm/rocAL/tree/develop/docs/examples/pytorch/toynet_training/train.py


.. |pipeline.py| replace:: ``pipeline.py``
.. _pipeline.py: https://github.com/ROCm/rocAL/tree/develop/tests/python_api/pipeline.py


