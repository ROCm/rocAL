/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "augmentations/node_uniform_distribution.h"

#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

UniformDistributionNode::UniformDistributionNode(
    const std::vector<Tensor *> &inputs,
    const std::vector<Tensor *> &outputs)
    : Node(inputs, outputs) {}

void UniformDistributionNode::create_node() {
    if (_node) return;

    for (uint i = 0; i < _batch_size; i++) {
        update_param();
        _uniform_distribution_array[i] = _dist_uniform(_rngs[i]);
    }
    _outputs[0]->swap_handle(static_cast<void *>(_uniform_distribution_array));
}

void UniformDistributionNode::update_node() {
    for (uint i = 0; i < _batch_size; i++) {
        update_param();
        _uniform_distribution_array[i] = _dist_uniform(_rngs[i]);
    }
}

void UniformDistributionNode::update_param() {
    std::uniform_real_distribution<float> dist_uniform(_min, _max);
    _dist_uniform = dist_uniform;
}

void UniformDistributionNode::init(std::vector<float> &range) {
    _min = range[0];
    _max = range[1];

    if (_outputs.empty() || !_outputs[0])
        THROW("UniformDistributionNode: output tensor is not initialized");

    _mem_type = _outputs[0]->info().mem_type();
    if (_mem_type == RocalMemType::HIP) {
#if ENABLE_HIP
        hipError_t err = hipHostMalloc(reinterpret_cast<void **>(&_uniform_distribution_array),
                                       _batch_size * sizeof(float),
                                       hipHostMallocDefault);
        if (err != hipSuccess || !_uniform_distribution_array)
            THROW("hipHostMalloc of size " + TOSTR(_batch_size * sizeof(float)) + " failed " + TOSTR(err))
#else
        THROW("UniformDistributionNode: GPU memory requested but HIP support (ENABLE_HIP) is disabled")
#endif
    } else {
        _uniform_distribution_array = new float[_batch_size];
    }

    BatchRNG<std::mt19937> rng = {ParameterFactory::instance()->get_seed_from_seedsequence(), static_cast<int>(_batch_size)};
    _rngs = rng;
    update_param();
}

UniformDistributionNode::~UniformDistributionNode() {
    if (!_uniform_distribution_array)
        return;

    if (_mem_type == RocalMemType::HIP) {
#if ENABLE_HIP
        hipError_t err = hipHostFree(_uniform_distribution_array);
        if (err != hipSuccess)
            std::cerr << "\n[ERR] hipHostFree failed for uniform distribution " << std::to_string(err) << "\n";
#else
        std::cerr << "\n[ERR] UniformDistributionNode: HIP memory requested but ENABLE_HIP is disabled\n";
#endif
    } else {
        delete[] _uniform_distribution_array;
    }

    _uniform_distribution_array = nullptr;
}
