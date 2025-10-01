.. meta::
  :description: rocAL Python APU
  :keywords: rocAL, ROCm, API, Python

********************************************************************
rocAL Python data types
********************************************************************

The rocAL data types are defined in `amd.rocal.types <https://github.com/ROCm/rocAL/blob/master/rocAL_pybind/amd/rocal/types.py>`_. 

Some of the commonly used rocAL data types are:

* Processing modes: Values (GPU/CPU). Use the rocal_cpu argument in the pipeline to set the processing mode. 

   * rocal_cpu = True: This performs data loading on the CPU. If GPUs are heavily used for training, it is viable to create the data-loading pipeline using CPU.
   * rocal_cpu = False: This performs data loading on the available GPU as specified using the device_id argument in the pipeline.

* Tensor output types: Values (NCHW/NHWC). Example: 

   * tensor_layout = types.NCHW
   * tensor_layout = types.NHWC

* Tensor data types: Values (FLOAT/FLOAT16). Example: 

   * tensor_dtype = types.FLOAT
   * tensor_dtype = types.FLOAT16

To see the usage of the above-mentioned data types, see `<https://github.com/ROCm/rocAL/blob/master/rocAL_pybind/amd/rocal/pipeline.py#L97>`__.

.. code-block:: python

    def __init__(self, batch_size=-1, num_threads=-1, device_id=-1, seed=-1,
                 exec_pipelined=True, prefetch_queue_depth=2,
                 exec_async=True, bytes_per_sample=0,
                 rocal_cpu=False, max_streams=-1, default_cuda_stream_priority=0, tensor_layout = types.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0], tensor_dtype=types.FLOAT):
        if(rocal_cpu):
            # print("comes to cpu")
            self._handle = b.rocalCreate(
                batch_size, types.CPU, device_id, num_threads,prefetch_queue_depth,types.FLOAT)
        else:
            print("comes to gpu")
            self._handle = b.rocalCreate(
                batch_size, types.GPU, device_id, num_threads,prefetch_queue_depth,types.FLOAT)  

