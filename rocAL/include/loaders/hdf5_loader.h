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

#pragma once

#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "loaders/circular_buffer.h"
#include "loaders/loader_module.h"
#include "pipeline/timing_debug.h"

class Hdf5Loader : public LoaderModule {
   public:
    explicit Hdf5Loader(void* device_resources);
    ~Hdf5Loader() override;

    void initialize(ReaderConfig reader_config, DecoderConfig decoder_config,
                    RocalMemType mem_type, unsigned batch_size,
                    bool keep_orig_size = false) override;
    void set_output(Tensor* output_tensor) override;
    void set_outputs(const std::vector<Tensor*>& output_tensors);
    LoaderModuleStatus load_next() override;
    void reset() override;
    void rethrow_if_error() override;
    size_t remaining_count() override;
    Timing timing() override;
    std::vector<std::string> get_id() override;
    DecodedDataInfo get_decode_data_info() override;
    void start_loading() override;
    void set_prefetch_queue_depth(size_t prefetch_queue_depth) override;
    void shut_down() override;
    size_t last_batch_padded_size() override;
    void feed_external_input(const std::vector<std::string>&, const std::vector<unsigned char*>&,
                             const std::vector<ROIxywh>&, unsigned, unsigned, unsigned,
                             ExternalSourceFileMode, bool) override {
        THROW("external source input is not supported for the HDF5 loader")
    }

   private:
    void stop_internal_thread();
    LoaderModuleStatus load_routine();
    void prepare_file_order(ReaderConfig& reader_config);

    ReaderConfig _reader_config{StorageType::HDF5_DATA};
    size_t _epoch = 0;
    void* _device_resources = nullptr;
    std::vector<Tensor*> _output_tensors;
    std::vector<std::unique_ptr<CircularBuffer>> _output_buffers;
    std::vector<size_t> _sample_sizes;
    std::vector<std::vector<uint32_t>> _sample_shapes;
    std::vector<std::string> _dataset_keys;
    std::vector<std::string> _files;
    mutable std::mutex _mutex;
    std::condition_variable _changed;
    std::exception_ptr _worker_error;
    size_t _queued_batches = 0;
    std::thread _load_thread;
    TimingDbg _file_load_time{"HDF5 file load time", DBG_TIMING};
    TimingDbg _swap_handle_time{"HDF5 swap handle time", DBG_TIMING};
    RocalMemType _mem_type = RocalMemType::HOST;
    size_t _batch_size = 1;
    size_t _prefetch_queue_depth = 3;
    size_t _next_file = 0;
    size_t _remaining_file_count = 0;
    size_t _last_batch_padded_size = 0;
    bool _loop = false;
    bool _initialized = false;
    bool _internal_thread_running = false;
    bool _stopped = false;
};
