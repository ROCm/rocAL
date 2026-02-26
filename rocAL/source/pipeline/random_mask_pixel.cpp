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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "pipeline/random_mask_pixel.h"

#include <algorithm>
#include <omp.h>

#include "parameters/parameter_factory.h"

RandomMaskPixel::RandomMaskPixel(size_t user_batch_size, size_t cpu_num_threads)
    : _user_batch_size(user_batch_size), _cpu_num_threads(cpu_num_threads) {}

void RandomMaskPixel::set_config(bool is_foreground, int value, bool is_threshold) {
    _is_foreground = is_foreground;
    _pixel_value = value;
    _is_threshold = is_threshold;
}

void RandomMaskPixel::ensure_rngs() {
    const unsigned seed = ParameterFactory::instance()->get_seed();
    if (_rng_seed == seed && _rngs.size() == _user_batch_size)
        return;
    _rng_seed = seed;
    _rngs.resize(_user_batch_size);
    for (size_t i = 0; i < _user_batch_size; i++) {
        std::seed_seq seq{seed, 0x4D504958u, static_cast<unsigned>(i)};  // "MPIX"
        _rngs[i].seed(seq);
    }
}

int64_t RandomMaskPixel::find_pixel(const std::vector<int> &start, const std::vector<int> &foreground_count, int64_t val, int count) {
    if (val < 0 || val >= count || start.empty() || foreground_count.empty() || start.size() != foreground_count.size())
        return -1;
    auto it = std::upper_bound(foreground_count.begin(), foreground_count.end(), val);
    if (it == foreground_count.begin())
        return -1;
    size_t idx = static_cast<size_t>(it - foreground_count.begin() - 1);
    return start[idx] + (val - foreground_count[idx]);
}

// Select a random pixel per sample from the batch of segmentation masks.
// When _is_foreground is false, picks a uniformly random pixel.
// When _is_foreground is true, builds run-length encoded foreground spans
// (using either threshold or equality matching), then uniformly samples
// among foreground pixels. Falls back to a uniformly random pixel if no
// foreground pixels exist in the mask.
TensorList *RandomMaskPixel::run(rocalTensorList *input, TensorList &output_list) {
    ensure_rngs();
    _output_buffer.clear();
    _output_buffer.resize(_user_batch_size * 2);
    const int nthreads = static_cast<int>(std::max<size_t>(1, std::min(_cpu_num_threads, _user_batch_size)));

    if (!_is_foreground) {
#pragma omp parallel for num_threads(nthreads)
        for (unsigned i = 0; i < _user_batch_size; i++) {
            auto &rng = _rngs[i];
            auto width = input->at(i)->dims().at(0);
            auto height = input->at(i)->dims().at(1);
            auto row = std::uniform_int_distribution<int64_t>(0, height - 1)(rng);
            auto col = std::uniform_int_distribution<int64_t>(0, width - 1)(rng);
            _output_buffer[i * 2] = row;
            _output_buffer[i * 2 + 1] = col;
        }
    } else {
        auto process_foreground = [&](unsigned i, auto match_fn) {
            std::vector<int> start;
            std::vector<int> foreground_count;
            unsigned id = 0;
            int count = 0;
            auto &rng = _rngs[i];
            int *mask_buffer = static_cast<int *>(input->at(i)->buffer());
            auto width = input->at(i)->dims().at(0);
            auto height = input->at(i)->dims().at(1);
            auto buffer_size = width * height;
            while (id < buffer_size) {
                if (!match_fn(mask_buffer[id])) {
                    id++;
                } else {
                    start.push_back(id++);
                    foreground_count.push_back(count++);
                    while (id < buffer_size && match_fn(mask_buffer[id])) {
                        id++;
                        count++;
                    }
                }
            }
            if (count != 0) {
                auto dist = std::uniform_int_distribution<int64_t>(0, count - 1);
                auto flat_idx = find_pixel(start, foreground_count, dist(rng), count);
                if (flat_idx >= 0) {
                    _output_buffer[i * 2] = flat_idx / width;
                    _output_buffer[i * 2 + 1] = flat_idx % width;
                    return;
                }
            }
            // Fallback: random pixel
            auto row = std::uniform_int_distribution<int64_t>(0, height - 1)(rng);
            auto col = std::uniform_int_distribution<int64_t>(0, width - 1)(rng);
            _output_buffer[i * 2] = row;
            _output_buffer[i * 2 + 1] = col;
        };

        if (_is_threshold) {
            int threshold = _pixel_value;
#pragma omp parallel for num_threads(nthreads)
            for (unsigned i = 0; i < _user_batch_size; i++) {
                process_foreground(i, [threshold](int val) { return val > threshold; });
            }
        } else {
            int target = _pixel_value;
#pragma omp parallel for num_threads(nthreads)
            for (unsigned i = 0; i < _user_batch_size; i++) {
                process_foreground(i, [target](int val) { return val == target; });
            }
        }
    }

    auto random_data_buffers = reinterpret_cast<unsigned int *>(_output_buffer.data());
    auto random_tensor_dims = {(size_t)2};
    for (unsigned i = 0; i < _user_batch_size; i++) {
        output_list[i]->set_dims(random_tensor_dims);
        output_list[i]->set_mem_handle(static_cast<void *>(random_data_buffers));
        random_data_buffers += 2;
    }

    return &output_list;
}
