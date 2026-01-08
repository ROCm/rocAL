.. meta::
  :description: rocAL documentation and API reference library
  :keywords: rocAL, ROCm, API, documentation

********************
The rocAL pipeline
********************

rocAL pipelines are used to load, decode, and augment images and video that will be used in training and inference. 

Video and image data is passed through the pipeline in batches. The next batch is prefetched and loaded while the initial batch is being processed. 

Pipelines are created from graph definition functions written by the user that have been decorated with ``@pipeline_def``. The ``@pipeline_def`` decorator converts a graph definition function into a pipeline factory.

Graph definition functions need to load a file, decode it, and augment it. The return value is the result of the augmentation.

For example, the ``my_pipe()`` function defines a graph that flips an image: 

.. code:: python

    @pipeline_def
    def my_pipe(flip_vertical, flip_horizontal):
        data, _ = fn.readers.file(file_root=images_dir)
        img = fn.decoders.image(data, device="mixed")
        flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
        return flipped, img

For more information on the ``@pipeline_def`` decorator, see the :doc:`pipeline API reference <../doxygen/html/pipeline_8py>`.

In addition to the graph, parameters such as the batch size, device ID, and tensor layout need to be set to build the pipeline. For example:

.. code:: python

    pipe = my_pipe(True, False, batch_size=32, num_threads=1, device_id=0)

See the :doc:`pipeline API reference <../doxygen/html/pipeline_8py>` for more information on the parameters that are needed to build a pipeline.

Once these parameters are set, the pipeline can be built and run:

.. code:: python

    pipe.build()
    flipped, img = pipe.run()

The output of the pipeline is the output of the original function.

``pipeline.run()`` doesn't need to be called explicitly if an iterator is being used. The rocAL iterators call ``pipeline.rocal_run()`` internally, eliminating the need to call ``pipeline.run()``.
