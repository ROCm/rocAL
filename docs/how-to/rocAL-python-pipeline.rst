.. meta::
  :description: rocAL Python APU
  :keywords: rocAL, ROCm, API, Python

*************************************************
Building and running a rocAL pipeline in Python
*************************************************

A rocAL pipeline is composed of multiple operations connected in an ordered graph encapsulated in an ``amd.rocal.pipeline`` object. ``amd.rocal.pipeline`` is a single library that can be integrated to build preprocessing pipelines for both training and inference applications. 

After the pipeline is instantiated, rocAL operators, such as file readers, are added to it. 

To define a pipeline
.. code-block:: shell
   :caption: Pipeline Class

    class Pipeline(object):


    Parameters
    ----------
    `batch_size` : int, optional, default = -1
        Batch size of the pipeline. Negative values for this parameter
        are invalid - the default value may only be used with
        serialized pipeline (the value stored in serialized pipeline
        is used instead).
    `num_threads` : int, optional, default = -1
        Number of CPU threads used by the pipeline.
        Negative values for this parameter are invalid - the default
        value may only be used with serialized pipeline (the value
        stored in serialized pipeline is used instead).
    `device_id` : int, optional, default = -1
        Id of GPU used by the pipeline.
        Negative values for this parameter are invalid - the default
        value may only be used with serialized pipeline (the value
        stored in serialized pipeline is used instead).
    `seed` : int, optional, default = -1
        Seed used for random number generation. Leaving the default value
        for this parameter results in random seed.
    `exec_pipelined` : bool, optional, default = True
        Whether to execute the pipeline in a way that enables
        overlapping CPU and GPU computation, typically resulting
        in faster execution speed, but larger memory consumption.
    `prefetch_queue_depth` : int or {"cpu_size": int, "gpu_size": int}, optional, default = 2
        Depth of the executor pipeline. Deeper pipeline makes ROCAL
        more resistant to uneven execution time of each batch, but it
        also consumes more memory for internal buffers.
        Specifying a dict:
        ``{ "cpu_size": x, "gpu_size": y }``
        instead of an integer will cause the pipeline to use separated
        queues executor, with buffer queue size `x` for cpu stage
        and `y` for mixed and gpu stages. It is not supported when both `exec_async`
        and `exec_pipelined` are set to `False`.
        Executor will buffer cpu and gpu stages separately,
        and will fill the buffer queues when the first :meth:`amd.rocal.pipeline.Pipeline.run`
        is issued.
    `exec_async` : bool, optional, default = True
        Whether to execute the pipeline asynchronously.
        This makes :meth:`amd.rocal.pipeline.Pipeline.run` method
        run asynchronously with respect to the calling Python thread.
        In order to synchronize with the pipeline one needs to call
        :meth:`amd.rocal.pipeline.Pipeline.outputs` method.
    `bytes_per_sample` : int, optional, default = 0
        A hint for ROCAL for how much memory to use for its tensors.
    `set_affinity` : bool, optional, default = False
        Whether to set CPU core affinity to the one closest to the
        GPU being used.
    `max_streams` : int, optional, default = -1
        Limit the number of HIP streams used by the executor.
        Value of -1 does not impose a limit.
        This parameter is currently unused (and behavior of
        unrestricted number of streams is assumed).
    `default_cuda_stream_priority` : int, optional, default = 0
        HIP stream priority used by ROCAL. 


    pipe = SimplePipeline(batch_size=batch_size, num_threads=num_threads, device_id=args.local_rank, seed=random_seed, rocal_cpu=rocal_cpu, tensor_layout=types.NHWC if args.NHWC else types.NCHW , tensor_dtype=types.FLOAT16 if args.fp16 else types.FLOAT)





    # Set Params
    output_set = 0
    rocal_device = 'cpu' if rocal_cpu else 'gpu'
    decoder_device = 'cpu' if rocal_cpu else 'gpu'
    # Use pipeline instance to make calls to reader, decoder & augmentations 
    with pipe:
        jpegs, _ = fn.readers.file(file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        images = fn.decoders.image(jpegs, file_root=data_path, device=decoder_device, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=True)
        images = fn.resize(images, device=rocal_device, resize_x=300, resize_y=300)

Defining the Pipeline
------------------------


Following are the important functions available in the Pipeline class, which is an instance of ``amd.rocal.pipeline``:

* ``build()``: Used to build a pipeline graph
* ``__init__ constructor``: Defines all the operators to be used in the graph with the corresponding parameters
* ``is_empty()``: Used to check if all the pipeline handles are empty
* ``rocalResetLoaders()``: Used to reset the iterator to the beginning
* ``set_outputs()``: Used to set the augmentations output of the graph

Building the Pipeline
-------------------------

Building the pipeline ensures that all operators are validated with the corresponding inputs and outputs.

To build the pipeline, see `https://github.com/ROCm/rocAL/blob/master/tests/python_api/unit_test.py#L166`

.. code-block:: python
   :caption: Build the Pipeline

    # build the pipeline
    pipe = SimplePipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
    pipe.build()


Running the Pipeline
-----------------------------

To run/use the pipeline, simply create a data loader using the pipeline and iterate through it to get the next batch of images with labels.

To run the pipeline, see `<https://github.com/ROCm/rocAL/blob/master/tests/python_api/unit_test.py#L168>`__.

.. code-block:: python
   :caption: Run the Pipeline

    # Dataloader
    data_loader = ROCALClassificationIterator(pipe,device=device)
    # Enumerate over the Dataloader
    for epoch in range(int(args.num_epochs)):
        print("EPOCH:::::", epoch)
        for i, it in enumerate(data_loader, 0):


Pipeline Output 
-------------------------

The output of the pipeline created above for 4 iterations (number of epochs) with a batch size of 2 is shown below for your reference. Each image is decoded and resized to 224x224.

.. figure:: ../data/ch4_sample.png

   Sample Pipeline Output


Performing Augmentations
================================

rocAL not only reads images from the disk and batches them into tensors, it can also perform various augmentations on those images. 

To read images, decode them, and rotate them in the pipeline, see `<https://github.com/ROCm/rocAL/blob/master/tests/python_api/unit_test.py#L77>`__

.. code-block:: python
   :caption: Perform Augmentations

    def rotated_pipeline():
        jpegs, labels = fn.readers.file(file_root=image_dir, random_shuffle=True)
        images = fn.decoders.image(jpegs, device='cpu')

    # Rotate the decoded images at an angle of 10ᵒ and fill the remaining space
    With black color (0)
        rotated_images = fn.rotate(images, angle=10.0, fill_value=0)
        return rotated_images, labels

    pipe = rotated_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
    pipe.build()


To run the pipeline, see:

.. code-block:: python

    pipe_out = pipe.run()
    images, labels = pipe_out
    show_images(images)