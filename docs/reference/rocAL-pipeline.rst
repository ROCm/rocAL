.. meta::
  :description: rocAL documentation and API reference library
  :keywords: rocAL, ROCm, API, documentation

********************
The rocAL pipeline
********************

rocAL pipelines are used to load, decode, and augment audio, video, and image files that will be used in training and inference. 

Audio, video, and image data is passed through the pipeline in batches. 

Pipelines can be defined by decorating a graph definition function with ``@pipeline_def`` or by using the ``Pipeline()`` constructor.

The ``@pipeline_def`` decorator converts a graph definition function into a pipeline factory. The graph definition function needs to load a file, decode it, and :doc:`augment it <../conceptual/rocAL-operators>`. The return value is the result of the augmentation.

For example, the ``my_pipe()`` function defines a graph that flips an image: 

.. code:: python

    from amd.rocal.pipeline import pipeline_def

    [...]

    @pipeline_def
    def my_pipe(flip_vertical, flip_horizontal):
        data, _ = fn.readers.file(file_root=images_dir)
        img = fn.decoders.image(data, device="mixed")
        flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
        return flipped, img

Parameters such as the batch size, device ID, and tensor layout need to be set before the pipeline can be built. For example:

.. code:: python

    pipe = my_pipe(True, False, batch_size=32, num_threads=1, device_id=0)

For more information on creating a pipeline with ``@pipeline_def``, see :doc:`Creating and running a pipeline <../how-to/rocAL-use-pipeline>`.

When the pipeline is created with ``Pipeline()``, the pipeline must be populated with readers and augmentations. For example, in |train.py|_ the training pipeline is populated with a graph that decodes, resizes, flips, and crop-mirror-normalizes images:

.. code:: python

    from amd.rocal.pipeline import Pipeline
    
    [...]

    def trainPipeline(data_path, batch_size, num_classes, one_hot, local_rank, world_size, num_thread, crop, rocal_cpu, fp16):
        pipe = Pipeline(batch_size=batch_size, num_threads=num_thread, device_id=local_rank, seed=local_rank+10, 
                rocal_cpu=rocal_cpu, tensor_dtype = types.FLOAT16 if fp16 else types.FLOAT, tensor_layout=types.NCHW, 
                prefetch_queue_depth = 7)
        with pipe:
            jpegs, labels = fn.readers.file(file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
            rocal_device = 'cpu' if rocal_cpu else 'gpu'
            decode = fn.decoders.image_slice(jpegs, output_type=types.RGB, file_root=data_path, shard_id=local_rank, 
                    num_shards=world_size, random_shuffle=True)
            res = fn.resize(decode, resize_x=224, resize_y=224)
            flip_coin = fn.random.coin_flip(probability=0.5)
            cmnp = fn.crop_mirror_normalize(res, device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mirror=flip_coin,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
            if(one_hot):
                _ = fn.one_hot(labels, num_classes)
            pipe.set_outputs(cmnp)
        return pipe

In both cases, the pipeline must be built with ``pipe.build()`` before being run.

There are two ways to run a pipeline. The first is using the ``pipeline.run()`` function. This function will run the pipeline exactly once on a single batch of files. The second way is to use an iterator that prefetches and loads the next batch of files while the initial batch is being processed. 

.. code:: python

    pipe.build()
    data_loader = ROCALClassificationIterator(pipe, device=device)
    images = next(iter(data_loader))

The output of the pipeline is the output of the graph definition function.

.. |train.py| replace:: ``train.py``
.. _train.py: https://github.com/ROCm/rocAL/tree/develop/docs/examples/pytorch/toynet_training/train.py

.. |imagenet.py| replace:: ``imagenet_training.py``
.. _imagenet.py: https://github.com/ROCm/rocAL/tree/develop/docs/examples/pytorch/imagenet_training/imagenet_training.py

.. |pipeline.py| replace:: ``pipeline.py``
.. _pipeline.py: https://github.com/ROCm/rocAL/tree/develop/tests/python_api/pipeline.py

