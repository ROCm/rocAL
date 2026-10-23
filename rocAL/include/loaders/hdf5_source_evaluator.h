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

#include <mutex>

std::mutex& hdf5_api_mutex();

#include <string>
#include <vector>

#include "pipeline/commons.h"

struct Hdf5DatasetInfo {
    std::string key;
    std::vector<size_t> max_shape;
    RocalTensorDataType data_type;
};

struct Hdf5SourceInfo {
    std::vector<std::string> files;
    std::vector<Hdf5DatasetInfo> datasets;
};

// Inspects an HDF5 file set before graph construction. Dataset information is
// returned in dataset_keys order so the graph can create deterministic output
// tensors for an atomic multi-dataset reader.
class Hdf5SourceEvaluator {
   public:
    Hdf5SourceInfo evaluate(const std::string& source_path,
                            const std::vector<std::string>& files,
                            const std::vector<std::string>& dataset_keys) const;
};
