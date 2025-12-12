/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "augmentations/node_python_function.h"
#include "pipeline/exception.h"

#ifdef ROCAL_PYTHON_FUNCTION
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vx_ext_rpp.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>

namespace py = pybind11;

PythonFunctionNode::PythonFunctionNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
    : Node(inputs, outputs) {
}

void PythonFunctionNode::create_node() {
    if (_node)
        return;

    vx_context vx_ctx = vxGetContext((vx_reference)_graph->get());

    // Passing the function ID as a vx_scalar
    vx_scalar function_id_vx = vxCreateScalar(vx_ctx, VX_TYPE_UINT64, &_function_id);

    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());

    vx_scalar input_layout_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &output_layout);
    uint64_t bridge_fn_ptr = reinterpret_cast<uint64_t>(&rocal_process_python_function);
    vx_scalar bridge_fn_ptr_vx = vxCreateScalar(vx_ctx, VX_TYPE_UINT64, &bridge_fn_ptr);

    _node = vxExtPythonFunction(
        _graph->get(),
        _inputs[0]->handle(),
        _outputs[0]->handle(),
        bridge_fn_ptr_vx,
        function_id_vx,
        input_layout_vx,
        output_layout_vx);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS) {
        THROW("Adding the PythonFunction node failed: " + TOSTR(status));
    }
}

void PythonFunctionNode::init(unsigned long long function_id) {
    _function_id = function_id;
}

void PythonFunctionNode::update_node() {}

std::pair<std::string, size_t> numpy_type_from_vx(vx_enum type) {
    switch (type) {
        case VX_TYPE_FLOAT32:
            return {py::format_descriptor<float>::format(), sizeof(float)};
        case VX_TYPE_FLOAT16:
            return {std::string("e"), 2};  // NumPy float16
        case VX_TYPE_UINT8:
            return {py::format_descriptor<uint8_t>::format(), sizeof(uint8_t)};
        case VX_TYPE_INT8:
            return {py::format_descriptor<int8_t>::format(), sizeof(int8_t)};
        case VX_TYPE_INT16:
            return {py::format_descriptor<int16_t>::format(), sizeof(int16_t)};
        case VX_TYPE_UINT32:
            return {py::format_descriptor<uint32_t>::format(), sizeof(uint32_t)};
        case VX_TYPE_INT32:
            return {py::format_descriptor<int32_t>::format(), sizeof(int32_t)};
        default:
            throw std::runtime_error("Unsupported OpenVX dtype in rocal_process_python_function");
    }
}

vx_status rocal_process_python_function(void* src_ptr, void* dst_ptr, const RocalPyExecParams* params) {
    if (!src_ptr || !dst_ptr || !params)
        return VX_ERROR_INVALID_REFERENCE;

    // CPU-only for now
    if (params->device_type == AGO_TARGET_AFFINITY_GPU)
        return VX_ERROR_NOT_IMPLEMENTED;

    try {
        py::gil_scoped_acquire acquire;

        // Resolve input/output numpy dtype and itemsize
        auto input_np = numpy_type_from_vx(params->in_desc.dtype);
        auto output_np = numpy_type_from_vx(params->out_desc.dtype);
        const size_t output_itemsize = output_np.second;

        // Build shape/strides (in bytes) for input view
        const size_t input_ndim = params->in_desc.num_dims;
        const size_t output_ndim = params->out_desc.num_dims;

        std::vector<ssize_t> input_shape(input_ndim);
        std::vector<ssize_t> input_strides(input_ndim);
        for (size_t i = 0; i < input_ndim; ++i) {
            input_shape[i] = static_cast<ssize_t>(params->in_desc.shape[i]);
            input_strides[i] = static_cast<ssize_t>(params->in_desc.strides[i] * input_np.second);
        }

        // Zero-copy NumPy view over src_ptr
        py::capsule owner(src_ptr, [](void*) { /* no-op: memory owned by caller */ });
        py::array input_numpy_batch(
            py::dtype(input_np.first),
            input_shape,
            input_strides,
            src_ptr,
            owner);

        // Reconstruct Python callable from function_id
        py::handle fh(reinterpret_cast<PyObject*>(params->function_id));
        py::object python_function = py::reinterpret_borrow<py::object>(fh);

        // Validate that the object is callable
        if (!PyCallable_Check(python_function.ptr())) {
            ERR("Object is not callable\n");
            return VX_ERROR_INVALID_REFERENCE;
        }

        // Call the python function
        py::object result_obj = python_function(input_numpy_batch);

        // Ensure python returned a NumPy array
        if (!py::isinstance<py::array>(result_obj)) {
            ERR(std::string("Python function did not return a NumPy array. Got: ") +
                std::string(py::str(result_obj.get_type())));
            return VX_ERROR_INVALID_TYPE;
        }

        // Safe typed wrapper (no conversion) since we've already validated it is an ndarray
        py::array result_array = py::reinterpret_borrow<py::array>(result_obj);

        // Ensure contiguous result for memcpy
        py::array result_contig = py::array::ensure(result_array, py::array::c_style);
        if (!result_contig) {
            ERR("Failed to obtain a C-contiguous array from Python result.");
            return VX_ERROR_INVALID_TYPE;
        }

        // Validate output against out_desc
        py::buffer_info buf = result_contig.request();
        if (buf.ndim != static_cast<int>(output_ndim)) {
            ERR(std::string("Dimension mismatch - expected ") + std::to_string(output_ndim) +
                " dimensions, got " + std::to_string(buf.ndim));
            return VX_ERROR_INVALID_DIMENSION;
        }
        // Compare shape
        for (size_t i = 0; i < output_ndim; ++i) {
            const size_t expected = params->out_desc.shape[i];
            const size_t got = static_cast<size_t>(buf.shape[i]);
            if (expected != got) {
                ERR(std::string("Shape mismatch at dimension ") + std::to_string(i) +
                    " - expected " + std::to_string(expected) +
                    ", got " + std::to_string(got));
                return VX_ERROR_INVALID_DIMENSION;
            }
        }

        // Verify returned array dtype matches expected dtype
        py::dtype expected_dtype = py::dtype(output_np.first);
        py::dtype got_dtype = result_contig.dtype();

        // Portable dtype equality check
        bool dtype_ok = py::bool_(got_dtype.attr("__eq__")(expected_dtype));
        if (!dtype_ok) {
            ERR(std::string("Dtype mismatch - expected ") + std::string(py::str(expected_dtype)) +
                ", got " + std::string(py::str(got_dtype)));
            return VX_ERROR_INVALID_TYPE;
        }

        // Calculate expected destination buffer size
        size_t dst_total_bytes = output_itemsize;
        for (size_t i = 0; i < output_ndim; ++i)
            dst_total_bytes *= params->out_desc.shape[i];
        std::memcpy(dst_ptr, buf.ptr, dst_total_bytes);

    } catch (const py::error_already_set& e) {
        // Python exception occurred
        ERR("Python error: " + std::string(e.what()) + "\n");
        return VX_FAILURE;
    } catch (const std::exception& e) {
        ERR("std::exception: " + std::string(e.what()) + "\n");
        return VX_FAILURE;
    } catch (...) {
        ERR("Unknown exception\n");
        return VX_FAILURE;
    }

    return VX_SUCCESS;
}
#endif
