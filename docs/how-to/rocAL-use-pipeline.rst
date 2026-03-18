.. meta::
  :description: rocAL pipeline
  :keywords: rocAL, ROCm, API, pipeline, decorator

**************************************
Creating and running a rocAL pipeline
**************************************

rocAL pipelines are used to load, decode, and augment audio, video, and image files that will be used in training and inference. 

The pipeline can either be instantiated using the ``Pipeline()`` constructor or by using the ``@pipeline_def`` decorator. This document demonstrates how to use the decorator. For information about using the constructor, see :doc:`the rocAL pipeline reference <../reference/rocAL-pipeline>`.

To create and use a pipeline in your rocAL application with the ``@pipeline_def`` decorator, you'll need to import ``@pipeline_def`` from ``amd.rocal.pipeline``:

.. code:: python

  from amd.rocal.pipeline import pipeline_def

There are two ways to run a pipeline. The first is using the ``pipeline.run()`` function. This function will run the pipeline exactly once on one batch of files. The second way is to use an iterator that runs all batches of files through the pipeline.

Audio, video, and image iterators are available, with some iterators designed to work with specific framework integrations. For example, if you're using :doc:`PyTorch for training <./rocAL-pytorch-framework>`, you would import the ``ROCALClassificationIterator`` in ``amd.rocal.plugin.pytorch``:

.. code:: python

  from amd.rocal.plugin.pytorch import ROCALClassificationIterator

Generic iterators are imported from ``amd.rocal.plugin.generic``:

.. code:: python

  from amd.rocal.plugin.generic import ROCALClassificationIterator


A pipeline is created by decorating a graph definition function with ``@pipeline_def``. 
 
Graph definition functions are user-defined functions that import audio, video, or image files, decode them, and augment them. The ``@pipeline_def`` decorator turns a graph definition function into a pipeline factory. The output of the graph definition function becomes the output of the pipeline.

For example, in |decoder.py|_ the graph definition function, ``image_decoder_pipeline``, reads in an image file, decodes it, and resizes it. It then returns the resized image:

.. code:: python

  @pipeline_def(seed=seed)
  def image_decoder_pipeline(device="cpu", path=image_dir):
    jpegs, labels = fn.readers.file(file_root=path)
    images = fn.decoders.image(jpegs, file_root=path, device=device, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
    return fn.resize(images, device=device, resize_width=300, resize_height=300)

The pipeline object requires additional parameters such as batch size, number of threads, and device ID. These are passed to the decorated function. 

For example, in |decoder.py|:

.. code:: python

  pipe = image_decoder_pipeline(batch_size=bs, num_threads=1, device_id=gpu_id, rocal_cpu=rocal_cpu, tensor_layout=types.NHWC, reverse_channels=True, mean = [0, 0, 0], std=[255,255,255], device=rocal_device, path=img_folder)

See the :doc:`pipeline API reference <../doxygen/html/pipeline_8py>` for the complete list of parameters.

Once the pipeline is created, ``pipeline.build()`` is called to build the pipeline. The pipeline can only be run after it's been built.

Use an iterator to run the pipeline over every batch of files. In |decoder.py|, the pipeline is run using ``ROCALClassificationIterator``. 

.. code:: python

  def show_pipeline_output(pipe, device):
    pipe.build()
    data_loader = ROCALClassificationIterator(pipe, device=device)
    images = next(iter(data_loader))
    show_images(images[0][0])

  [...]

  def main():
    [...]
    pipe = image_decoder_pipeline(batch_size=bs, num_threads=1, device_id=gpu_id, rocal_cpu=rocal_cpu, tensor_layout=types.NHWC,reverse_channels=True, mean = [0, 0, 0], std=[255,255,255], device=rocal_device, path=img_folder)
    show_pipeline_output(pipe, device=rocal_device)

The iterator will run the pipeline until all batches of files have been processed.

.. |pipeline.py| replace:: ``pipeline.py``
.. _pipeline.py: https://github.com/ROCm/rocAL/tree/develop/tests/python_api/pipeline.py

.. |decoder.py| replace:: ``decoder.py``
.. _decoder.py: https://github.com/ROCm/rocAL/tree/develop/tests/python_api/decoder.py