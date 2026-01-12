.. meta::
  :description: rocAL documentation and API reference library
  :keywords: rocAL, ROCm, API, documentation

**************************************
Creating and running a rocAL pipeline
**************************************

rocAL pipelines are used to load, decode, and augment audio, video, and image files that will be used in training and inference. 

To create and use a pipeline in your rocAL application, you'll need to import ``@pipeline_def`` from ``amd.rocal.pipeline``:

.. code:: python

  from amd.rocal.pipeline import pipeline_def

Iterators also need to be imported. If you're using :doc:`PyTorch for training <./rocAL-pytorch-framework>`, use the ``ROCALClassificationIterator`` in ``amd.rocal.plugin.pytorch``:

.. code:: python

  from amd.rocal.plugin.pytorch import ROCALClassificationIterator

Otherwise, use the generic iterator, ``ROCALClassificationIterator``, in ``amd.rocal.plugin.generic``:

.. code:: python

  from amd.rocal.plugin.generic import ROCALClassificationIterator

A pipeline is created by decorating a graph definition function with ``@pipeline_def``. 
 
Graph definition functions are user-defined functions that import audio, video, and image files, decode them, and augment them. The ``@pipeline_def`` decorator turns a graph definition function into a pipeline factory. The output of the graph definition function becomes the output of the pipeline.

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

  pipe = image_decoder_pipeline(batch_size=bs, num_threads=1, device_id=gpu_id, rocal_cpu=rocal_cpu, tensor_layout=types.   NHWC, reverse_channels=True, mean = [0, 0, 0], std=[255,255,255], device=rocal_device, path=img_folder)

See the :doc:`pipeline API reference <../doxygen/html/pipeline_8py>` for the complete list of parameters.

Once the pipeline is created, ``pipeline.build()`` is called to build the pipeline before the pipeline is run. 

The ``pipeline.run()`` function can be used to explicitly run the pipeline or the pipeline can be run through an iterator.

For example, in |decoder.py| the pipeline is built and run in the ``show_pipeline_output()`` function. The ``build()`` function is called explicitly to build the pipeline. The pipeline is then run as part of the backend of the ``ROCALClassificationIterator``:

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

The pipeline runs until all the input files have been processed.

.. |inference_pipeline.py| replace:: ``decoder.py``
.. _inference_pipeline.py: https://github.com/ROCm/rocAL/tree/develop/docs/examples/image_processing/inference_pipeline.py

.. |decoder.py| replace:: ``decoder.py``
.. _decoder.py: https://github.com/ROCm/rocAL/tree/develop/tests/python_api/decoder.py