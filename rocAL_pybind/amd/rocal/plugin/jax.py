# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##
# @file jax.py
#
# @brief File containing iterators and functions for JAX framework

import jax
import jax.dlpack
import jax.numpy as jnp
from jax.sharding import NamedSharding, PositionalSharding

try:
    from clu.data.dataset_iterator import ArraySpec
    CLU_FOUND = True
except ImportError:
    CLU_FOUND = False

import threading
import concurrent.futures
import numpy as np
from packaging.version import Version

import rocal_pybind as b
import amd.rocal.types as types

if Version(jax.__version__) < Version("0.4.23"):
    print('rocAL only supports jax versions >= 0.4.23')


def convert_to_jax_array(array):
    """Converts input DLPack tensor to JAX array.

    Args:
        array (DLPack tensor):
            array to be converted to JAX array.

    Returns:
        jax.Array: JAX array with the same values and device as input array.
    """
    jax_array = jax.dlpack.from_dlpack(array, copy=True)
    return jax_array

class ROCALJaxIterator(object):
    """Initializes the ROCAL JAX iterator.

    Args:
        pipelines (list of Pipeline objects): List of rocAL pipelines to use.
        sharding (JAX sharding, optional): JAX sharding to use for placing outputs on devices. Defaults to None.
    """

    def __init__(self, pipelines, sharding=None):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        self.pipelines = pipelines
        self.num_devices = len(pipelines)
        self.batch_size = pipelines[0]._batch_size

        self.iterator_length = b.getRemainingImages(
            pipelines[0]._handle) // self.batch_size  # Length should be the same across all pipelines
        self.last_batch_policy = pipelines[0]._last_batch_policy
        assert (
            self.last_batch_policy != types.LAST_BATCH_PARTIAL
        ), "JAX iterator does not support partial last batch policy."

        if sharding is not None:
            assert isinstance(
                sharding, (NamedSharding, PositionalSharding)
            ), "`sharding` should be an instance of `NamedSharding` or `PositionalSharding`"
        self.sharding = sharding
        self._has_started = False

    def next(self):
        """Returns the next batch of data."""
        return self.__next__()

    def __next__(self):
        """Returns the next batch of data."""
        self._has_started = True
        pipeline_outputs = []
        for pipeline in self.pipelines:
            if pipeline.rocal_run() != 0:
                raise StopIteration
            output_tensor_list = pipeline.get_output_tensors()

            output_list = []
            device_id = pipeline._device_id
            if pipeline._name is None:
                pipeline._name = pipeline._reader
            labels_size = ((self.batch_size * pipeline._num_classes)
                                if pipeline._one_hot_encoding else self.batch_size)
            for i in range(len(output_tensor_list)):
                output = convert_to_jax_array(
                    output_tensor_list[i].__dlpack__(device_id))
                output_list.append(output)

            if pipeline._name == "labelReader":
                if pipeline._one_hot_encoding:
                    labels = np.empty(labels_size, dtype="int32")
                    pipeline.get_one_hot_encoded_labels(
                        labels.ctypes.data, pipeline._output_memory_type)
                    labels_tensor = labels.reshape(
                        -1, self.batch_size, pipeline._num_classes)
                    labels_tensor = convert_to_jax_array(
                        labels_tensor)
                else:
                    labels = pipeline.get_image_labels()
                    labels_tensor = labels.astype(dtype=np.int32)
                    labels_tensor = convert_to_jax_array(
                        labels_tensor)
                labels_tensor = jax.device_put(
                    labels_tensor, output_list[0].device)
                output_list.append(labels_tensor)
            pipeline_outputs.append(output_list)

        if self.num_devices == 1 and self.sharding is None:
            return pipeline_outputs[0]

        sharded_outputs = []
        for i in range(len(pipeline_outputs[0])):
            individual_outputs = []
            for pipeline_id in range(self.num_devices):
                individual_outputs.append(pipeline_outputs[pipeline_id][i])
            for output in individual_outputs:
                assert output.shape == individual_outputs[0].shape, "All outputs should have the same shape"
            if self.sharding is not None:
                sharded_outputs.append(self.place_output_with_sharding(
                    individual_outputs))
            else:
                sharded_outputs.append(self.place_output_with_device_put(
                    individual_outputs))
        return sharded_outputs

    def reset(self):
        """Resets the iterator for the next epoch."""
        for pipeline in self.pipelines:
            b.rocalResetLoaders(pipeline._handle)

    def place_output_with_device_put(self, individual_outputs):
        """Builds sharded jax.Array with `jax.device_put_sharded` - compatible
        with pmapped JAX functions.
        """
        output_devices = tuple(
            map(lambda jax_shard: jax_shard.device, individual_outputs)
        )

        if len(output_devices) != len(set(output_devices)):
            if len(set(output_devices)) != 1:
                raise AssertionError(
                    (
                        "JAX iterator requires shards to be placed on different devices "
                        "or all on the same device."
                    )
                )
            else:
                # All shards are on one device (CPU or one GPU)
                return jnp.stack(individual_outputs)
        else:
            return jax.device_put_sharded(individual_outputs, output_devices)

    def place_output_with_sharding(self, individual_outputs):
        """Builds sharded jax.Array with `jax.make_array_from_single_device_arrays`-
        compatible with automatic parallelization with JAX.
        """
        shard_shape = individual_outputs[0].shape
        data_rank = len(shard_shape)
        num_devices = len(individual_outputs)

        # Calculate the global shape. The first (batch) dimension is aggregated.
        global_shape = (num_devices * shard_shape[0],) + shard_shape[1:]

        if isinstance(self.sharding, NamedSharding):
            # Assumes the user wants to shard along the first mesh axis
            # and replicate across the other data dimensions.
            mesh = self.sharding.mesh
            # Create a PartitionSpec matching the data's rank.
            # e.g., for 4D data and mesh axis 'data', this becomes P('data', None, None, None)
            partition_spec = jax.sharding.PartitionSpec(
                mesh.axis_names[0], *(None for _ in range(data_rank - 1)))
            compatible_sharding = NamedSharding(mesh, partition_spec)
        else:
            # Assumes the user wants to shard along the single device axis
            # and replicate across the other data dimensions.
            devices = self.sharding._devices
            # Create a sharding shape matching the data's rank.
            # e.g., for 4D data and 8 devices, this becomes (8, 1, 1, 1)
            sharding_spec_shape = (num_devices,) + (1,) * (data_rank - 1)
            compatible_sharding = PositionalSharding(devices).reshape(sharding_spec_shape)

        return jax.make_array_from_single_device_arrays(
            global_shape, compatible_sharding, individual_outputs
        )

    def __iter__(self):
        """Returns the iterator object."""
        return self

    def __len__(self):
        """Returns the number of batches in the iterator."""
        return self.iterator_length

    def __del__(self):
        """Releases the rocAL resources."""
        for pipeline in self.pipelines:
            b.rocalRelease(pipeline._handle)


def get_spec_for_array(jax_array):
    """Returns the ArraySpec for a given JAX array.

    Args:
        jax_array (jax.Array): The JAX array.

    Returns:
        ArraySpec: The specification of the array.
    """
    return ArraySpec(shape=jax_array.shape, dtype=jax_array.dtype)


class ROCALPeekableIterator(ROCALJaxIterator):
    """ROCALJaxIterator extended with peek functionality. Compatible with Google CLU PeekableIterator.

     Reference: https://github.com/google/CommonLoopUtils/blob/main/clu/data/dataset_iterator.py
    """

    def __init__(self, pipelines, sharding=None):
        if not CLU_FOUND:
            print('Install CLU for peekable data iterator support')
            raise ImportError
        super().__init__(
            pipelines,
            sharding
        )
        self.mutex = threading.Lock()
        self.pool = None
        self.peek = None

        self.element_spec = None

    def set_element_spec(self, outputs):
        """Sets the element spec for the iterator.

        Args:
            outputs: The output from the iterator.
        """
        self.element_spec = [get_spec_for_array(output) for output in outputs]

    def assert_output_shape_and_type(self, outputs):
        """Asserts that the shape and type of the outputs are consistent.

        Args:
            outputs: The output from the iterator.

        Raises:
            ValueError: If the shape or type of the output changes between iterations.

        Returns:
            The outputs if they are consistent.
        """
        if self.element_spec is None:
            # Set element spec based on the first seen element
            self.set_element_spec(outputs)

        for idx, _ in enumerate(outputs):
            if get_spec_for_array(outputs[idx]) != self.element_spec[idx]:
                raise ValueError(
                    "The shape or type of the output changed between iterations. "
                    "This is not supported by JAX  peekable iterator. "
                    "Please make sure that the shape and type of the output is constant. "
                    f"Expected: {self.element_spec[idx]}, got: {get_spec_for_array(outputs[idx])} "
                    f"for output: {idx}"
                )

        return outputs

    def next_with_peek_impl(self):
        """Returns the next element from the iterator and advances the iterator.
        """
        if self.peek is None:
            return self.assert_output_shape_and_type(super().__next__())
        peek = self.peek
        self.peek = None
        return self.assert_output_shape_and_type(peek)

    def __next__(self):
        """Returns the next element from the iterator and advances it."""
        with self.mutex:
            return self.next_with_peek_impl()

    def __iter__(self):
        """Returns the iterator object. Resets if the iterator has been used."""
        if self._has_started and self.peek is None:
            self.reset()
        return self

    def peek(self):
        """Returns the next element from the iterator without advancing the iterator.
        """
        with self.mutex:
            if self.peek is None:
                self.peek = self.next_with_peek_impl()
            return self.peek

    def peek_async(self):
        """Returns future that will return the next element from
        the iterator without advancing the iterator.
        """
        if self.pool is None:
            # Create pool only if needed (peek_async is ever called)
            # to avoid thread creation overhead
            self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        future = self.pool.submit(self.peek)
        return future

    @property
    def element_spec(self):
        """Returns the element spec for the elements returned by the iterator.
        """
        if self.element_spec is None:
            self.set_element_spec(self.peek())
        return self.element_spec
