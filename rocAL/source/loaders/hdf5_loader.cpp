/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "loaders/hdf5_loader.h"

#include <algorithm>
#include <random>

#include "pipeline/exception.h"
#include "loaders/hdf5_source_evaluator.h"
#include <limits>

#if defined(ROCAL_HDF5)
#include <H5Cpp.h>
#endif

namespace {
#if defined(ROCAL_HDF5)
const H5::PredType& native_hdf5_type(RocalTensorDataType data_type) {
    switch (data_type) {
        case RocalTensorDataType::FP32: return H5::PredType::NATIVE_FLOAT;
        case RocalTensorDataType::UINT8: return H5::PredType::NATIVE_UINT8;
        case RocalTensorDataType::UINT32: return H5::PredType::NATIVE_UINT32;
        case RocalTensorDataType::INT32: return H5::PredType::NATIVE_INT32;
        case RocalTensorDataType::INT16: return H5::PredType::NATIVE_INT16;
        default: THROW("HDF5 loader encountered an unsupported rocAL tensor data type")
    }
}
#endif
}  // namespace

Hdf5Loader::Hdf5Loader(void* device_resources) : _device_resources(device_resources) {}

Hdf5Loader::~Hdf5Loader() {
    shut_down();
}

void Hdf5Loader::set_output(Tensor* output_tensor) {
    set_outputs({output_tensor});
}

void Hdf5Loader::set_outputs(const std::vector<Tensor*>& output_tensors) {
    if (output_tensors.empty())
        THROW("HDF5 loader requires at least one output tensor")
    _output_tensors = output_tensors;
}

void Hdf5Loader::prepare_file_order(ReaderConfig& reader_config) {
    const auto all_files = reader_config.get_files();
    const size_t shard_count = reader_config.get_shard_count();
    const auto& policy = reader_config.get_sharding_info();
    if (all_files.empty() || shard_count == 0 || reader_config.get_shard_id() >= shard_count)
        THROW("HDF5 loader received invalid files or shard arguments")
    if (shard_count > all_files.size())
        THROW("HDF5 reader requires at least one file per shard")
    if (policy.shard_size != -1 && policy.shard_size <= 0)
        THROW("HDF5 shard_size must be -1 or a positive number")
    if (_loop && policy.last_batch_policy == RocalBatchPolicy::PARTIAL)
        THROW("HDF5 PARTIAL batches cannot be combined with loop")

    const size_t shard_id = (reader_config.get_shard_id() +
                            (policy.stick_to_shard ? 0 : _epoch % shard_count)) % shard_count;
    _files.clear();
    for (size_t index = shard_id; index < all_files.size(); index += shard_count)
        _files.push_back(all_files[index]);
    if (reader_config.shuffle()) {
        std::mt19937 random_engine(reader_config.seed() + static_cast<unsigned>(_epoch));
        std::shuffle(_files.begin(), _files.end(), random_engine);
    }
    size_t smallest = all_files.size() / shard_count;
    size_t largest = smallest + (all_files.size() % shard_count != 0);
    if (policy.shard_size > 0) {
        smallest = std::min(smallest, static_cast<size_t>(policy.shard_size));
        largest = std::min(largest, static_cast<size_t>(policy.shard_size));
        if (_files.size() > static_cast<size_t>(policy.shard_size))
            _files.resize(static_cast<size_t>(policy.shard_size));
    }
    auto batches = [this, &policy](size_t size) {
        return size / _batch_size +
               (policy.last_batch_policy == RocalBatchPolicy::PARTIAL && size % _batch_size != 0);
    };
    if (policy.last_batch_policy != RocalBatchPolicy::FILL && batches(smallest) != batches(largest))
        THROW("HDF5 shards would have unequal batch counts; use FILL or a smaller shard_size")

    _last_batch_padded_size = 0;
    if (policy.last_batch_policy == RocalBatchPolicy::DROP) {
        _files.resize((_files.size() / _batch_size) * _batch_size);
    } else {
        const size_t valid_size = _files.size();
        const size_t target_size = policy.last_batch_policy == RocalBatchPolicy::FILL ? largest : valid_size;
        const size_t padded_size = target_size + (_batch_size - target_size % _batch_size) % _batch_size;
        _last_batch_padded_size = padded_size - valid_size;
        while (_files.size() < padded_size) {
            const size_t index = policy.pad_last_batch_repeated || policy.last_batch_policy == RocalBatchPolicy::PARTIAL
                                     ? valid_size - 1 : _files.size() % valid_size;
            _files.push_back(_files[index]);
        }
    }
    if (_files.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        THROW("HDF5 shard exceeds the rocAL sample-count limit")
    if (_loop && _files.empty())
        THROW("HDF5 loop requires at least one complete batch")
}

void Hdf5Loader::initialize(ReaderConfig reader_config, DecoderConfig,
                            RocalMemType mem_type, unsigned batch_size, bool) {
#if !defined(ROCAL_HDF5)
    (void)reader_config;
    (void)mem_type;
    (void)batch_size;
    THROW("rocAL was built without HDF5 reader support")
#else
    if (_initialized)
        THROW("HDF5 loader is already initialized")
    if (_output_tensors.empty())
        THROW("HDF5 loader outputs must be set before initialization")
    _dataset_keys = reader_config.get_dataset_keys();
    if (_dataset_keys.size() != _output_tensors.size())
        THROW("HDF5 loader dataset key count must match output tensor count")
    if (batch_size == 0)
        THROW("HDF5 loader batch size must be greater than zero")

    _batch_size = batch_size;
    _mem_type = mem_type;
    _loop = reader_config.loop();
    _reader_config = reader_config;
    prepare_file_order(_reader_config);
    _remaining_file_count = _files.size();
    _decoded_data_info._data_names.resize(_batch_size);
    _decoded_data_info._roi_shape.resize(_batch_size);

    _output_buffers.clear();
    _sample_sizes.clear();
    _sample_shapes.clear();
    for (auto* output : _output_tensors) {
        if (!output)
            THROW("HDF5 loader received a null output tensor")
        const auto output_size = output->info().data_size();
        if (output_size == 0 || output_size % _batch_size != 0)
            THROW("HDF5 loader output tensor has an invalid size")
        _sample_sizes.push_back(output_size / _batch_size);
        const auto max_shape = output->info().max_shape();
        _sample_shapes.emplace_back(max_shape.begin(), max_shape.end());
        auto buffer = std::make_unique<CircularBuffer>(_device_resources);
        try {
            buffer->init(_mem_type, output_size, _prefetch_queue_depth);
        } catch (...) {
            buffer->release();
            throw;
        }
        _output_buffers.emplace_back(std::move(buffer));
    }
    _initialized = true;
#endif
}

void Hdf5Loader::start_loading() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_initialized)
        THROW("HDF5 loader must be initialized before loading starts")
    if (_load_thread.joinable())
        THROW("HDF5 loader is already running")
    _stopped = false;
    _worker_error = nullptr;
    _internal_thread_running = true;
    _load_thread = std::thread(&Hdf5Loader::load_routine, this);
}

LoaderModuleStatus Hdf5Loader::load_routine() {
#if !defined(ROCAL_HDF5)
    return LoaderModuleStatus::NOT_INITIALIZED;
#else
    std::string file_path, dataset_key;
    try {
#if ENABLE_HIP
        if (_mem_type == RocalMemType::HIP) {
            if (!_device_resources)
                THROW("HDF5 loader has no HIP device resources")
            const int device_id = static_cast<DeviceResourcesHip*>(_device_resources)->device_id;
            const auto status = hipSetDevice(device_id);
            if (status != hipSuccess)
                THROW("hipSetDevice failed in Hdf5Loader::load_routine: " + TOSTR(status))
        }
#endif
        while (true) {
            std::vector<unsigned char*> write_buffers;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _changed.wait(lock, [this] {
                    return !_internal_thread_running || _queued_batches < _prefetch_queue_depth - 1;
                });
                if (!_internal_thread_running)
                    break;
                if (!_loop && (_files.size() - _next_file) < _batch_size) {
                    _internal_thread_running = false;
                    _changed.notify_all();
                    break;
                }
                for (auto& buffer : _output_buffers) {
                    auto* destination = buffer->get_write_buffer();
                    if (!destination)
                        THROW("HDF5 loader could not allocate an output buffer")
                    write_buffers.push_back(destination);
                }
                _file_load_time.start();
            }

            DecodedDataInfo batch_info;
            batch_info._data_names.resize(_batch_size);
            for (size_t batch_index = 0; batch_index < _batch_size; ++batch_index) {
                if (_next_file >= _files.size())
                    _next_file = 0;
                file_path = _files[_next_file++];
                dataset_key.clear();
                std::lock_guard<std::mutex> hdf5_lock(hdf5_api_mutex());
                H5::Exception::dontPrint();
                H5::H5File input_file(file_path, H5F_ACC_RDONLY);
                for (size_t output_index = 0; output_index < _dataset_keys.size(); ++output_index) {
                    dataset_key = _dataset_keys[output_index];
                    auto dataset = input_file.openDataSet(dataset_key);
                    auto* destination = write_buffers[output_index] +
                                        batch_index * _sample_sizes[output_index];
                    const auto& info = _output_tensors[output_index]->info();
                    const auto& dims = info.dims();
                    std::vector<hsize_t> expected(dims.begin() + 1, dims.end());
                    auto file_space = dataset.getSpace();
                    if (file_space.getSimpleExtentNdims() != static_cast<int>(expected.size()))
                        THROW("dataset rank changed after schema inspection")
                    std::vector<hsize_t> actual(expected.size());
                    file_space.getSimpleExtentDims(actual.data());
                    if (actual != expected)
                        THROW("dataset shape changed after schema inspection")
                    const auto& native_type = native_hdf5_type(info.data_type());
                    auto type = dataset.getDataType();
                    if (type.getClass() != native_type.getClass() || type.getSize() != native_type.getSize())
                        THROW("dataset type changed after schema inspection")
                    if (type.getClass() == H5T_INTEGER &&
                        H5::IntType(type.getId()).getSign() != H5::IntType(native_type.getId()).getSign())
                        THROW("dataset signedness changed after schema inspection")
                    H5::DataSpace memory_space(static_cast<int>(expected.size()), expected.data());
                    dataset.read(destination, native_type, memory_space, file_space);
                }
                batch_info._data_names[batch_index] = file_path;
            }
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _file_load_time.end();
                if (!_internal_thread_running)
                    break;
                // Publish only after every dataset in the batch has been read.
                for (auto& buffer : _output_buffers) {
                    buffer->set_decoded_data_info(batch_info);
                    buffer->push();
                }
                ++_queued_batches;
            }
            _changed.notify_all();
        }
    } catch (...) {
        std::exception_ptr error;
        const auto location = file_path.empty() ? std::string("HDF5 worker initialization failed: ")
            : "HDF5 read failed in file '" + file_path + "', dataset '" + dataset_key + "': ";
        try {
            throw;
        } catch (const H5::Exception& cause) {
            error = std::make_exception_ptr(std::runtime_error(location + cause.getDetailMsg()));
        } catch (const std::exception& cause) {
            error = std::make_exception_ptr(std::runtime_error(location + cause.what()));
        } catch (...) {
            error = std::make_exception_ptr(std::runtime_error(location + "unknown error"));
        }
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _worker_error = error;
            _internal_thread_running = false;
        }
        _changed.notify_all();
        return LoaderModuleStatus::DECODE_FAILED;
    }
    return LoaderModuleStatus::OK;
#endif
}

void Hdf5Loader::rethrow_if_error() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_worker_error)
        std::rethrow_exception(_worker_error);
}

LoaderModuleStatus Hdf5Loader::load_next() {
    std::unique_lock<std::mutex> lock(_mutex);
    _changed.wait(lock, [this] {
        return _worker_error || _stopped || _queued_batches || !_internal_thread_running;
    });
    if (_worker_error)
        std::rethrow_exception(_worker_error);
    if (_stopped)
        return LoaderModuleStatus::OK;
    if (!_queued_batches)
        return LoaderModuleStatus::NO_MORE_DATA_TO_READ;

    try {
        _swap_handle_time.start();
        for (size_t index = 0; index < _output_tensors.size(); ++index) {
            void* buffer = _mem_type == RocalMemType::HIP
                               ? _output_buffers[index]->get_read_buffer_dev()
                               : static_cast<void*>(_output_buffers[index]->get_read_buffer_host());
            if (_output_tensors[index]->swap_handle(buffer) != 0)
                THROW("HDF5 loader could not swap an output tensor buffer")

            std::vector<std::vector<uint32_t>> batch_shapes(_batch_size, _sample_shapes[index]);
            _output_tensors[index]->update_tensor_roi(batch_shapes);
        }
        _swap_handle_time.end();
        _decoded_data_info = _output_buffers.front()->get_decoded_data_info();
        for (auto& buffer : _output_buffers)
            buffer->pop();
        --_queued_batches;
        if (!_loop)
            _remaining_file_count -= _batch_size;
    } catch (...) {
        _worker_error = std::current_exception();
        _internal_thread_running = false;
        _changed.notify_all();
        throw;
    }
    lock.unlock();
    _changed.notify_all();
    return LoaderModuleStatus::OK;
}

void Hdf5Loader::reset() {
    stop_internal_thread();
    for (auto& buffer : _output_buffers)
        buffer->reset();
    ++_epoch;
    prepare_file_order(_reader_config);
    _next_file = 0;
    _queued_batches = 0;
    _decoded_data_info = {};
    _remaining_file_count = _files.size();
    start_loading();
}

void Hdf5Loader::stop_internal_thread() {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _internal_thread_running = false;
        _stopped = true;
    }
    _changed.notify_all();
    if (_load_thread.joinable())
        _load_thread.join();
}

void Hdf5Loader::shut_down() {
    stop_internal_thread();
    for (auto& buffer : _output_buffers)
        buffer->release();
    _output_buffers.clear();
    _initialized = false;
}

size_t Hdf5Loader::remaining_count() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _remaining_file_count;
}
size_t Hdf5Loader::last_batch_padded_size() { return _last_batch_padded_size; }
std::vector<std::string> Hdf5Loader::get_id() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _decoded_data_info._data_names;
}
DecodedDataInfo Hdf5Loader::get_decode_data_info() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _decoded_data_info;
}

Timing Hdf5Loader::timing() {
    std::lock_guard<std::mutex> lock(_mutex);
    Timing timing;
    timing.read_time = _file_load_time.get_timing();
    timing.process_time = _swap_handle_time.get_timing();
    return timing;
}

void Hdf5Loader::set_prefetch_queue_depth(size_t prefetch_queue_depth) {
    if (prefetch_queue_depth < 2)
        THROW("HDF5 loader prefetch queue depth must be at least two")
    _prefetch_queue_depth = prefetch_queue_depth;
}
