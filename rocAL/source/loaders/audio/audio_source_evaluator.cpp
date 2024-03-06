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

#include "audio_source_evaluator.h"
#include "audio_decoder_factory.h"
#include "reader_factory.h"

size_t AudioSourceEvaluator::max_samples() {
    return _samples_max;
}

size_t AudioSourceEvaluator::max_channels() {
    return _channels_max;
}

AudioSourceEvaluatorStatus
AudioSourceEvaluator::create(ReaderConfig reader_cfg, DecoderConfig decoder_cfg) {
    AudioSourceEvaluatorStatus status = AudioSourceEvaluatorStatus::OK;
    // Can initialize it to any decoder types if needed
    _input_path = reader_cfg.path();
    if(_input_path.back() != '/') {
        _input_path = _input_path + "/";
    }
    _decoder = create_audio_decoder(std::move(decoder_cfg));
    _reader = create_reader(std::move(reader_cfg));
    find_max_dimension();
    return status;
}

void
AudioSourceEvaluator::find_max_dimension() {
    _reader->reset();
    while( _reader->count_items() ) {
        size_t fsize = _reader->open();
        if( (fsize) == 0 )
            continue;
        auto file_name = _reader->file_path();
        if(_decoder->initialize(file_name.c_str()) != AudioDecoder::Status::OK) {
            WRN("Could not initialize audio decoder for file : "+ _reader->id())
            continue;
        }
        int samples, channels;
        float sample_rates;
        if(_decoder->decode_info(&samples, &channels, &sample_rates) != AudioDecoder::Status::OK) {
            WRN("Could not decode the header of the: "+ _reader->id())
            continue;
        }
        if(samples <= 0 || channels <= 0)
            continue;
        _samples_max = std::max(samples, _samples_max);
        _channels_max = std::max(channels, _channels_max);
        _decoder->release();
    }
    // return the reader read pointer to the begining of the resource
    _reader->reset();
}


