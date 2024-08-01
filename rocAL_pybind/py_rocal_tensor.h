/*
Copyright (c) 2024 - Advanced Micro Devices, Inc. All rights reserved.

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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <dlpack/dlpack.h>
#include "rocal_api_types.h"
#include "rocal_api.h"
#include "rocal_api_tensor.h"
#include "rocal_api_parameters.h"
#include "rocal_api_data_loaders.h"
#include "rocal_api_augmentation.h"
#include "rocal_api_data_transfer.h"
#include "rocal_api_info.h"
namespace py = pybind11;

#define TENSOR_MAX_RANK 4

class PyRocalTensor {
    public:
        static void ExportUtil(py::module &m);
        py::capsule dlpack(py::object stream) const;
        py::tuple dlpack_device() const;
        static RocalTensor from_dlpack(py::object src, RocalTensorLayout elayout);
    private:
        PyRocalTensor() {}
        std::unique_ptr<rocalTensor> rocal_tensor_;
};
