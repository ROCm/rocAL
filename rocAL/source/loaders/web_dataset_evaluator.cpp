/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "loaders/web_dataset_evaluator.h"

#include "decoders/image/decoder_factory.h"
#include "readers/image/reader_factory.h"
void WebDataSetSourceEvaluator::set_size_evaluation_policy(WebDataSetMaxSizeEvaluationPolicy arg) {
    _width_max.set_policy(arg);
    _height_max.set_policy(arg);
}

size_t WebDataSetSourceEvaluator::max_width() {
    return _width_max.get_max();
}

size_t WebDataSetSourceEvaluator::max_height() {
    return _height_max.get_max();
}

WebDataSetSourceEvaluatorStatus
WebDataSetSourceEvaluator::create(ReaderConfig reader_cfg, DecoderConfig decoder_cfg) {
    WebDataSetSourceEvaluatorStatus status = WebDataSetSourceEvaluatorStatus::OK;
    _decoder = create_decoder(std::move(decoder_cfg));
    _reader = create_reader(std::move(reader_cfg));
    find_max_dimension();
    return status;
}

void WebDataSetSourceEvaluator::find_max_dimension() {
    _reader->reset();
    while (_reader->count_items()) {
        size_t fsize = _reader->open();
        if ((fsize) == 0)
            continue;
        _header_buff.resize(fsize);
        auto actual_read_size = _reader->read_data(_header_buff.data(), fsize);
        _reader->close();

        int width, height, jpeg_sub_samp;
        if (_decoder->decode_info(_header_buff.data(), actual_read_size, &width, &height, &jpeg_sub_samp) != Decoder::Status::OK) {
            WRN("Could not decode the header of the: " + _reader->id())
            continue;
        }

        if (width <= 0 || height <= 0)
            continue;

        _width_max.process_sample(width);
        _height_max.process_sample(height);
    }
    _reader->reset();
}

void WebDataSetSourceEvaluator::FindMaxSize::process_sample(unsigned val) {
    if (_policy == WebDataSetMaxSizeEvaluationPolicy::MAXIMUM_FOUND_SIZE) {
        _max = (val > _max) ? val : _max;
    }
    if (_policy == WebDataSetMaxSizeEvaluationPolicy::MOST_FREQUENT_SIZE) {
        auto it = _hist.find(val);
        size_t count = 1;
        if (it != _hist.end()) {
            it->second = +1;
            count = it->second;
        } else {
            _hist.insert(std::make_pair(val, 1));
        }

        unsigned new_count = count;
        if (new_count > _max_count) {
            _max = val;
            _max_count = new_count;
        }
    }
}