.. meta::
  :description: Using augmentations in a rocAL pipeline
  :keywords: rocAL, ROCm, API, augmentations, pipeline, how to

********************************************************************
Using augmentations in a rocAL pipeline
********************************************************************

Augmentations change the data in a pipeline, for example by resizing, cropping, or flipping images, or by changing their brightness or color. 
They add variety to a dataset and prepare data for training and inference. 
This topic explains how to add augmentations to a rocAL pipeline, how to set their parameters, and how to view their output. 
For a list of the augmentations that rocAL provides, see the :doc:`rocAL augmentations reference <../reference/rocAL-augmentations>`. 

Before you start, make sure you're familiar with how to :doc:`create and run a rocAL pipeline <./rocAL-use-pipeline>`.

Add an augmentation to a pipeline
=================================

Augmentations are in the ``amd.rocal.fn`` module. Import it together with the ``amd.rocal.types`` module, which defines the enumerations that some augmentation arguments use:

.. code:: python

  import amd.rocal.fn as fn
  import amd.rocal.types as types
  from amd.rocal.pipeline import pipeline_def

Call an augmentation inside the graph definition function of the pipeline. 
The first argument of an augmentation is the data to augment, which is the output of a decoder or of another augmentation. 
The augmentation returns the augmented data. 
In the following example, ``fn.brightness`` increases the brightness of the decoded images that were resized by ``fn.resize``. 

.. code:: python

  @pipeline_def(seed=seed)
  def augmentation_pipeline(device="cpu", path=image_dir):
    jpegs, labels = fn.readers.file(file_root=path)
    images = fn.decoders.image(jpegs, file_root=path, device=device, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
    resized = fn.resize(images, resize_width=300, resize_height=300)
    return fn.brightness(resized, brightness=1.5)

In this example and the examples that follow, ``seed`` is the random seed for the pipeline and ``image_dir`` is the path to the folder that contains the images. 
For example, the rocAL example notebooks use the images in ``/opt/rocm/share/rocal/test/data/images/AMD-tinyDataSet/``.

The ``brightness`` argument is a multiplier. A value of 0 gives a black image, 1 leaves the image unchanged, and 2 doubles the brightness.

To see the arguments that an augmentation accepts and their default values, see the :doc:`augmentations Python API reference <../doxygen/html/fn_8py>`.

Chain augmentations
===================

To apply more than one augmentation to the same images, pass the output of one augmentation to the next. The augmentations run in the order that you call them. 
For example, the following pipeline resizes the images, increases their brightness, and then flips them horizontally:

.. code:: python

  @pipeline_def(seed=seed)
  def chained_pipeline(device="cpu", path=image_dir):
    jpegs, labels = fn.readers.file(file_root=path)
    images = fn.decoders.image(jpegs, file_root=path, device=device, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
    resized = fn.resize(images, resize_width=300, resize_height=300)
    brighter = fn.brightness(resized, brightness=1.5)
    return fn.flip(brighter, horizontal=1)

Some augmentations combine several operations into one, which is faster than calling the operations separately. 
For example, ``fn.crop_mirror_normalize`` crops, flips, and normalizes the images in one operation, and ``fn.resize_crop_mirror`` resizes, crops, and flips them in one operation. 
The :doc:`rocAL augmentations reference <../reference/rocAL-augmentations>` describes these augmentations.

Get the output of more than one augmentation
============================================

To compare the effect of different augmentations, apply each augmentation to the same input and return all of the results. 
A graph definition function that returns a tuple creates a pipeline with one output for each item in the tuple:

.. code:: python

  @pipeline_def(seed=seed)
  def multi_output_pipeline(device="cpu", path=image_dir):
    jpegs, labels = fn.readers.file(file_root=path)
    images = fn.decoders.image(jpegs, file_root=path, device=device, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
    resized = fn.resize(images, resize_width=300, resize_height=300)
    brightness_output = fn.brightness(resized)
    contrast_output = fn.contrast(resized)
    fog_output = fn.fog(resized)
    return resized, brightness_output, contrast_output, fog_output

If you create the pipeline with the ``Pipeline()`` constructor instead of the ``@pipeline_def`` decorator, call ``pipe.set_outputs()`` with the outputs inside a ``with pipe:`` block. 
For an example, see the `augmentation examples notebook <https://github.com/ROCm/rocAL/blob/develop/docs/examples/notebooks/augmentation_examples.ipynb>`_.

Set augmentation parameters
===========================

You can pass a fixed value to an augmentation argument, or you can pass a random value generator so that each image in a batch gets a different value.

Fixed values
------------

Pass a number to use the same value for every image:

.. code:: python

  fn.brightness(resized, brightness=1.5)

Random values
-------------

To get a different value for each image, pass the output of a random value generator instead of a number. 
``fn.uniform`` generates random float values within a range. 
In the following example, the saturation, contrast, brightness, and hue of each image is set to a random value within the given range:

.. code:: python

  saturation = fn.uniform(range=[0.1, 0.4])
  contrast = fn.uniform(range=[0.1, 25.0])
  brightness = fn.uniform(range=[0.875, 1.125])
  hue = fn.uniform(range=[5.0, 170.0])
  augmented = fn.color_twist(resized, saturation=saturation, contrast=contrast, brightness=brightness, hue=hue)

``fn.random.coin_flip`` generates a 0 or a 1, with a given probability of getting a 1. 
Use it to apply an augmentation to only some of the images. 
For example, the following code flips about half of the images horizontally as they're cropped and normalized:

.. code:: python

  coin_flip = fn.random.coin_flip(probability=0.5)
  cmnp = fn.crop_mirror_normalize(resized,
                                  output_dtype=types.FLOAT,
                                  output_layout=types.NCHW,
                                  crop=(224, 224),
                                  mirror=coin_flip,
                                  mean=[0, 0, 0],
                                  std=[1, 1, 1])

The ``fn.random`` module also provides ``fn.random.uniform`` and ``fn.random.normal``, which create tensors from a uniform and a normal distribution. 
For more information, see the :doc:`rocAL augmentations reference <../reference/rocAL-augmentations>`.

Set the layout and data type of the output
==========================================

Most image augmentations have ``output_layout`` and ``output_dtype`` arguments that set the layout and data type of the augmentation output. 
The defaults are ``types.NHWC`` for the layout and ``types.UINT8`` for the data type. 
In the previous example, ``fn.crop_mirror_normalize`` outputs a ``types.FLOAT`` tensor in ``types.NCHW`` layout, which is the format that PyTorch models use. 

The ``Pipeline()`` constructor and the ``@pipeline_def`` decorator also accept the ``tensor_layout`` and ``tensor_dtype`` arguments, which set the layout and data type that the pipeline uses for its tensors. 
See the :doc:`pipeline API reference <../doxygen/html/pipeline_8py>` for the complete list of pipeline arguments.

Run the pipeline and view the output
====================================

Build the pipeline and use an iterator to run it, as described in :doc:`Create and run the pipeline <./rocAL-use-pipeline>`. 
The iterator returns the output of each of the augmentations in the order in which the graph definition function returns them. 

.. code:: python

  from amd.rocal.plugin.generic import ROCALClassificationIterator

  pipe = multi_output_pipeline(batch_size=1, num_threads=1, device_id=0, rocal_cpu=True, tensor_layout=types.NHWC, tensor_dtype=types.FLOAT)
  pipe.build()
  data_loader = ROCALClassificationIterator(pipe)

  for it in data_loader:
    for output in it[0]:
      # Each item is the output of one of the augmentations
      print(output.shape)
  data_loader.reset()

The `augmentation examples notebook <https://github.com/ROCm/rocAL/blob/develop/docs/examples/notebooks/augmentation_examples.ipynb>`_ shows how to display the output of each augmentation using ``matplotlib``. 

Choose the CPU or GPU backend
=============================

The backend that an augmentation runs on is set when the pipeline is created. Set ``rocal_cpu=True`` to use the CPU backend, or ``rocal_cpu=False`` (the default) to use the GPU backend. 
The ``device`` argument that most augmentations accept is unused.

Some augmentations are only available on one backend. For example, ``fn.color_jitter`` is supported on the CPU backend only, and ``fn.python_function`` runs on the host only and must be enabled when you build rocAL. 
The :doc:`rocAL augmentations reference <../reference/rocAL-augmentations>` notes these restrictions.
