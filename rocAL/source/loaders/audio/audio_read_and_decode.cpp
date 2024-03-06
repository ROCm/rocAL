/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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


#include <iterator>
#include <cstring>
#include "decoder_factory.h"
#include "audio_decoder_factory.h"
#include "audio_read_and_decode.h"

Timing
AudioReadAndDecode::timing() {
    Timing t;
    t.audio_decode_time = _decode_time.get_timing();
    t.audio_read_time = _file_load_time.get_timing();
    return t;
}

AudioReadAndDecode::AudioReadAndDecode():
    _file_load_time("FileLoadTime", DBG_TIMING ),
    _decode_time("DecodeTime", DBG_TIMING) {
}

AudioReadAndDecode::~AudioReadAndDecode() {
    _reader = nullptr;
    _decoder.clear();
}

void
AudioReadAndDecode::create(ReaderConfig reader_config, DecoderConfig decoder_config, int batch_size, int device_id) {
    // Can initialize it to any decoder types if needed
    _batch_size = batch_size;
    _decoder.resize(batch_size);
    _audio_names.resize(batch_size);
    _audio_file_path.resize(batch_size);
    _decompressed_buff_ptrs.resize(_batch_size);
    _actual_decoded_samples.resize(_batch_size);
    _actual_decoded_channels.resize(_batch_size);
    _original_channels.resize(_batch_size);
    _original_samples.resize(_batch_size);
    _original_sample_rates.resize(_batch_size);
    _decoder_config = decoder_config;
    if ((_decoder_config._type != DecoderType::SKIP_DECODE)) {
        for (int i = 0; i < batch_size; i++) {
            _decoder[i] = create_audio_decoder(decoder_config);
        }
    }
    _reader = create_reader(reader_config);
    _input_path = reader_config.path();
    _num_threads = reader_config.get_cpu_num_threads();
    if(_input_path.back() != '/')
        _input_path = _input_path + "/";
}

void
AudioReadAndDecode::reset() {
    _reader->reset();
}

size_t
AudioReadAndDecode::count() {
    return _reader->count_items();
}

// size_t
// AudioReadAndDecode::last_batch_padded_size() {
//     return _reader->last_batch_padded_size();
// }

LoaderModuleStatus
AudioReadAndDecode::load(float* buff,
                         std::vector<std::string>& names,
                         const size_t max_decoded_samples,
                         const size_t max_decoded_channels,
                         std::vector<uint32_t> &roi_samples,
                         std::vector<uint32_t> &roi_channels,
                         std::vector<float> &actual_sample_rates) {
    if(max_decoded_samples == 0 || max_decoded_channels == 0 )
        THROW("Zero audio dimension is not valid")
    if(!buff)
        THROW("Null pointer passed as output buffer")
    if(_reader->count_items() < _batch_size)
        return LoaderModuleStatus::NO_MORE_DATA_TO_READ;
    // load audios/frames from the disk and push them as a large audio onto the buff
    unsigned file_counter = 0;
    const size_t audio_size = max_decoded_samples * max_decoded_channels;
    // Decode with the channels and size equal to a single audio
    // File read is done serially since I/O parallelization does not work very well.
    _file_load_time.start();// Debug timing
    while ((file_counter != _batch_size) && _reader->count_items() > 0) {
        size_t fsize = _reader->open();
        if (fsize == 0) {
            WRN("Opened file " + _reader->id() + " of size 0");
            continue;
        }
        _audio_names[file_counter] = _reader->id();
        _audio_file_path[file_counter] = _reader->file_path();
        _reader->close();
        file_counter++;
    }
    _file_load_time.end();// Debug timing
    _decode_time.start();// Debug timing
    if (_decoder_config._type != DecoderType::SKIP_DECODE) {
        for (size_t i = 0; i < _batch_size; i++){
            _decompressed_buff_ptrs[i] = buff + (audio_size * i);
        }
#pragma omp parallel for num_threads(_num_threads)  // default(none) TBD: option disabled in Ubuntu 20.04
        for (size_t i = 0; i < _batch_size; i++) {
            // initialize the actual decoded channels and samples with the maximum
            _actual_decoded_samples[i] = max_decoded_samples;
            _actual_decoded_channels[i] = max_decoded_channels;
            int original_samples, original_channels;
            float original_sample_rates;
            if (_decoder[i]->initialize(_audio_file_path[i].c_str()) != AudioDecoder::Status::OK) {
                THROW("Decoder can't be initialized for file: " + _audio_names[i].c_str())
            }
            if (_decoder[i]->decode_info(&original_samples, &original_channels, &original_sample_rates) != AudioDecoder::Status::OK) {
                THROW("Unable to fetch decode info for file: " + _audio_names[i].c_str())
            }
            _original_channels[i] = original_channels;
            _original_samples[i] = original_samples;
            _original_sample_rates[i] = original_sample_rates;
            if (_decoder[i]->decode(_decompressed_buff_ptrs[i]) != AudioDecoder::Status::OK) {
                THROW("Decoder failed for file: " + _audio_names[i].c_str())
            }
            _decoder[i]->release();
        }
        for (size_t i = 0; i < _batch_size; i++) {
            names[i] = _audio_names[i];
            roi_samples[i] = _original_samples[i];
            roi_channels[i] = _original_channels[i];
            actual_sample_rates[i] = _original_sample_rates[i];
        }
    }
    _decode_time.end();// Debug timing
    return LoaderModuleStatus::OK;
}

