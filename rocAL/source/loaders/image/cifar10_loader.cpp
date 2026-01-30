/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "loaders/image/cifar10_loader.h"

#include <chrono>
#include <thread>

#include "vx_ext_amd.h"

#define NEAREST_MULTIPLE_OF_8(size) (((size) + 8) & ~7)

CIFAR10Loader::CIFAR10Loader(void* dev_resources) : _circ_buff(dev_resources),
                                                            _file_load_time("file load time", DBG_TIMING),
                                                            _swap_handle_time("Swap_handle_time", DBG_TIMING) {
    _output_tensor = nullptr;
    _mem_type = RocalMemType::HOST;
    _internal_thread_running = false;
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
    _remaining_image_count = 0;
    _device_id = 0;
}

CIFAR10Loader::~CIFAR10Loader() {
    de_init();
}

void CIFAR10Loader::set_prefetch_queue_depth(size_t prefetch_queue_depth) {
    if (prefetch_queue_depth <= 0)
        THROW("Prefetch queue depth value cannot be zero or negative");
    _prefetch_queue_depth = prefetch_queue_depth;
}

void CIFAR10Loader::set_gpu_device_id(int device_id) {
    if (device_id < 0)
        THROW("invalid device_id passed to loader");
    _device_id = device_id;
}

size_t
CIFAR10Loader::remaining_count() {
    return _remaining_image_count;
}

void CIFAR10Loader::reset() {
    // stop the writer thread and empty the internal circular buffer
    _internal_thread_running = false;
    _circ_buff.unblock_writer();

    if (_load_thread.joinable())
        _load_thread.join();

    // Emptying the internal circular buffer
    _circ_buff.reset();

    // resetting the reader thread to the start of the media
    _image_counter = 0;
    _reader->reset();

    // Start loading (writer thread) again
    start_loading();
}

void CIFAR10Loader::de_init() {
    stop_internal_thread();
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
    _remaining_image_count = 0;
}

void CIFAR10Loader::stop_internal_thread() {
    _internal_thread_running = false;
    _stopped = true;
    _circ_buff.unblock_reader();
    _circ_buff.unblock_writer();
    _circ_buff.reset();
    if (_load_thread.joinable())
        _load_thread.join();
}

LoaderModuleStatus
CIFAR10Loader::load_next() {
    return update_output_image();
}

void CIFAR10Loader::set_output(Tensor* output_tensor) {
    _output_tensor = output_tensor;
    // Align the output memory size to the nearest multiple of 8 bytes for efficient memory access
    _output_mem_size = NEAREST_MULTIPLE_OF_8(_output_tensor->info().data_size());
}

void CIFAR10Loader::initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RocalMemType mem_type, unsigned batch_size, bool keep_orig_size) {
    if (_is_initialized)
        WRN("initialize() function is already called and loader module is initialized")

    if (_output_mem_size == 0)
        THROW("output image size is 0, set_output_image() should be called before initialize for loader modules")
    // initialize loader and reader
    _mem_type = mem_type;
    _batch_size = batch_size;
    _loop = reader_cfg.loop();
    _image_size = _output_tensor->info().data_size() / batch_size;
    _output_names.resize(batch_size);
    try {
        _reader = create_reader(reader_cfg);
    } catch (const std::exception& e) {
        de_init();
        throw;
    }
    _actual_read_size.resize(batch_size);
    _decoded_data_info._data_names.resize(_batch_size);
    _decoded_data_info._roi_width.resize(_batch_size);  // used to store the individual image in a big raw file
    _decoded_data_info._roi_height.resize(batch_size);
    _decoded_data_info._original_height.resize(_batch_size);
    _decoded_data_info._original_width.resize(_batch_size);
    _crop_image_info._crop_image_coords.resize(_batch_size);
    _circ_buff.init(_mem_type, _output_mem_size, _prefetch_queue_depth);
    _is_initialized = true;
    LOG("Loader module initialized");
}

void CIFAR10Loader::set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader) {
    _randombboxcrop_meta_data_reader = randombboxcrop_meta_data_reader;
}

std::vector<std::vector<float>>&
CIFAR10Loader::get_batch_random_bbox_crop_coords() {
    return _crop_coords_batch;
}

void CIFAR10Loader::set_batch_random_bbox_crop_coords(std::vector<std::vector<float>> crop_coords) {
    _crop_coords_batch = crop_coords;
}

void CIFAR10Loader::shut_down() {
    if (_internal_thread_running)
        stop_internal_thread();
    _circ_buff.release();
}

void CIFAR10Loader::start_loading() {
    if (!_is_initialized)
        THROW("start_loading() should be called after initialize() function is called")

    _remaining_image_count = _reader->count_items();
    _internal_thread_running = true;
    _load_thread = std::thread(&CIFAR10Loader::load_routine, this);
}

LoaderModuleStatus
CIFAR10Loader::load_routine() {
    LOG("Started the internal loader thread");
    LoaderModuleStatus last_load_status = LoaderModuleStatus::OK;
    // Initially record number of all the images that are going to be loaded, this is used to know how many still there

    while (_internal_thread_running) {
        auto data = _circ_buff.get_write_buffer();
        auto cifar10reader = std::dynamic_pointer_cast<CIFAR10DataReader>(_reader);

        if (!_internal_thread_running)
            break;

        auto load_status = LoaderModuleStatus::NO_MORE_DATA_TO_READ;
        {
            unsigned file_counter = 0;
            _file_load_time.start();  // Debug timing

            while ((file_counter != _batch_size) && _reader->count_items() > 0) {
                auto read_ptr = data + _image_size * file_counter;
                size_t readSize = _reader->open();
                if (readSize == 0) {
                    ERR("Opened file " + _reader->id() + " of size 0");
                    continue;
                }
                _actual_read_size[file_counter] = _reader->read_data(read_ptr, readSize);
                _decoded_data_info._data_names[file_counter] = _reader->id();
                _decoded_data_info._roi_width[file_counter] = _output_tensor->info().max_shape()[0];
                _decoded_data_info._roi_height[file_counter] = _output_tensor->info().max_shape()[1];
                _reader->close();
                file_counter++;
            }
            if (_randombboxcrop_meta_data_reader) {
                // Fetch the crop co-ordinates for a batch of images
                _bbox_coords = _randombboxcrop_meta_data_reader->get_batch_crop_coords(_decoded_data_info._data_names);
                set_batch_random_bbox_crop_coords(_bbox_coords);
                _crop_image_info._crop_image_coords = get_batch_random_bbox_crop_coords();
                _circ_buff.set_crop_image_info(_crop_image_info);
            }
            _file_load_time.end();  // Debug timing
            _circ_buff.set_decoded_data_info(_decoded_data_info);
            _circ_buff.push();
            _image_counter += _output_tensor->info().batch_size();
            load_status = LoaderModuleStatus::OK;
        }
        if (load_status != LoaderModuleStatus::OK) {
            if (last_load_status != load_status) {
                if (load_status == LoaderModuleStatus::NO_MORE_DATA_TO_READ ||
                    load_status == LoaderModuleStatus::NO_FILES_TO_READ) {
                    LOG("Cycled through all images, count " + TOSTR(_image_counter));
                } else {
                    ERR("ERROR: Detected error in reading the images");
                }
                last_load_status = load_status;
            }

            // Here it sets the out-of-data flag and signal the circular buffer's internal
            // read semaphore using release() call
            // , and calls the release() allows the reader thread to wake up and handle
            // the out-of-data case properly
            // It also slows down the reader thread since there is no more data to read,
            // till program ends or till reset is called
            _circ_buff.unblock_reader();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    return LoaderModuleStatus::OK;
}

bool CIFAR10Loader::is_out_of_data() {
    return (remaining_count() < _batch_size);
}

size_t CIFAR10Loader::last_batch_padded_size() {
    return _reader->last_batch_padded_size();
}

LoaderModuleStatus
CIFAR10Loader::update_output_image() {
    LoaderModuleStatus status = LoaderModuleStatus::OK;

    if (is_out_of_data())
        return LoaderModuleStatus::NO_MORE_DATA_TO_READ;
    if (_stopped)
        return LoaderModuleStatus::OK;

    // _circ_buff.get_read_buffer_x() is blocking and puts the caller on sleep until new images are written to the _circ_buff
    if ((_mem_type == RocalMemType::OCL) || (_mem_type == RocalMemType::HIP)) {
        auto data_buffer = _circ_buff.get_read_buffer_dev();
        _swap_handle_time.start();
        if (_output_tensor->swap_handle(data_buffer) != 0)
            return LoaderModuleStatus ::DEVICE_BUFFER_SWAP_FAILED;
        _swap_handle_time.end();
    } else {
        auto data_buffer = _circ_buff.get_read_buffer_host();
        _swap_handle_time.start();
        if (_output_tensor->swap_handle(data_buffer) != 0)
            return LoaderModuleStatus::HOST_BUFFER_SWAP_FAILED;
        _swap_handle_time.end();
    }
    if (_stopped)
        return LoaderModuleStatus::OK;

    _output_decoded_data_info = _circ_buff.get_decoded_data_info();
    if (_randombboxcrop_meta_data_reader) {
        _output_cropped_image_info = _circ_buff.get_cropped_image_info();
    }
    _output_names = _output_decoded_data_info._data_names;
    _output_tensor->update_tensor_roi(_output_decoded_data_info._roi_width, _output_decoded_data_info._roi_height);

    _circ_buff.pop();
    if (!_loop)
        _remaining_image_count -= _batch_size;

    return status;
}

Timing CIFAR10Loader::timing() {
    Timing t;
    t.read_time = _file_load_time.get_timing();
    t.process_time = _swap_handle_time.get_timing();
    return t;
}

std::vector<std::string> CIFAR10Loader::get_id() {
    return _output_names;
}

DecodedDataInfo CIFAR10Loader::get_decode_data_info() {
    return _output_decoded_data_info;
}

CropImageInfo CIFAR10Loader::get_crop_image_info() {
    return _output_cropped_image_info;
}
