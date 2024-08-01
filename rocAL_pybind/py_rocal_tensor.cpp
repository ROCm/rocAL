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

#include "py_rocal_tensor.h"
namespace py = pybind11;

//dlpack functions
static bool supported_dl_device_type(const DLDeviceType &devType) {
    switch (devType) {
        case kDLROCM:
        case kDLROCMHost:
        case kDLCPU:
            return true;
        default:
            return false;
    }
}

static RocalTensorOutputType get_data_type(const DLDataType &dtype) {
    if (dtype.lanes != 1) {
        throw std::runtime_error("Data type lanes != 1 is not supported.");
    }

    switch (dtype.bits) {
        case 8:
            switch (dtype.code) {
                case kDLInt:
                    return RocalTensorOutputType::ROCAL_INT8;
                case kDLUInt:
                    return RocalTensorOutputType::ROCAL_UINT8;
                default:
                    throw std::runtime_error(
                        "Data type code is not supported.");
            }
        case 32:
            switch (dtype.code) {
                case kDLInt:
                    return RocalTensorOutputType::ROCAL_INT32;
                case kDLUInt:
                    return RocalTensorOutputType::ROCAL_UINT32;
                case kDLFloat:
                    return RocalTensorOutputType::ROCAL_FP32;
                default:
                    throw std::runtime_error(
                        "Data type code is not supported.");
            }
        case 16:
        default:
            throw std::runtime_error("Data type bits is not supported.");
    }
}

static DLDataType get_dl_data_type(const RocalTensorOutputType &dtype) {
    DLDataType out;
    out.lanes = 1;

    switch (dtype) {
        case RocalTensorOutputType::ROCAL_UINT8:
            out.bits = 8;
            out.code = kDLUInt;
            break;
        case RocalTensorOutputType::ROCAL_INT8:
            out.bits = 8;
            out.code = kDLInt;
            break;
        case RocalTensorOutputType::ROCAL_UINT32:
            out.bits = 32;
            out.code = kDLUInt;
            break;
        case RocalTensorOutputType::ROCAL_INT32:
            out.bits = 32;
            out.code = kDLInt;
            break;
        case RocalTensorOutputType::ROCAL_FP32:
            out.bits = 32;
            out.code = kDLFloat;
    }

    return out;
}

static RocalOutputMemType get_data_location(const DLDeviceType &devType) {
    switch (devType) {
        case kDLROCM:
            return RocalOutputMemType::ROCAL_MEMCPY_GPU;
        case kDLROCMHost:
        case kDLCPU:
            return RocalOutputMemType::ROCAL_MEMCPY_HOST;
        default:
            throw std::runtime_error("Tensor device type is not supported.");
    }
}

static DLDevice generate_dl_device(rocalTensor &rocal_tensor) {
    DLDevice dev;

    dev.device_id = 0;

    switch (rocal_tensor.mem_type()) {
        case RocalOutputMemType::ROCAL_MEMCPY_GPU:
            dev.device_type = kDLROCM;
            break;
        case RocalOutputMemType::ROCAL_MEMCPY_HOST:
            dev.device_type = kDLCPU;
            break;
    }
    std::cout << "generate_dl_device -- " << dev.device_type << std::endl;
    return dev;
}

py::tuple PyRocalTensor::dlpack_device() const {
    DLDevice dev = generate_dl_device(*this->rocal_tensor_);
    return py::make_tuple(py::int_(static_cast<int>(dev.device_type)),
                          py::int_(static_cast<int>(dev.device_id)));
}

py::capsule PyRocalTensor::dlpack(py::object stream) const{
    std::cout << "coming to dlpack function!!!" << std::endl;
    DLManagedTensor *dmtensor = new DLManagedTensor;
    dmtensor->deleter = [](DLManagedTensor *self) {
        delete[] self->dl_tensor.shape;
        delete[] self->dl_tensor.strides;
        delete self;
    };

    try {
        rocalTensor &rocal_tensor = *this->rocal_tensor_;
        DLTensor &dtensor = dmtensor->dl_tensor;

        dtensor.device = generate_dl_device(rocal_tensor);

        // Set up ndim
        dtensor.ndim = rocal_tensor.num_of_dims();

        // Set up data
        std::cout << "rocal buffer ptr -- " << rocal_tensor.buffer() << std::endl;
        dtensor.data = rocal_tensor.buffer();
        dtensor.byte_offset = 0;

        // Set up shape
        dtensor.shape = new int64_t[dtensor.ndim];
        std::vector<size_t> rocal_shape = rocal_tensor.shape();
        for (int32_t i = 0; i < dtensor.ndim; ++i) {
            dtensor.shape[i] = static_cast<int64_t>(rocal_shape[i]);
        }

        // Set up dtype
        dtensor.dtype = get_dl_data_type(rocal_tensor.data_type());

        // Set up strides
        dtensor.strides = new int64_t[dtensor.ndim];
        std::vector<size_t> rocal_strides = rocal_tensor.strides();
        for (int32_t i = 0; i < dtensor.ndim; i++) {
            int64_t stride = static_cast<int64_t>(rocal_strides[i]);
            if (stride % sizeof(rocal_tensor.data_type()) != 0) {
                throw std::runtime_error(
                    "Stride is not a multiple of the data type size.");
            }

            dtensor.strides[i] = stride / sizeof(rocal_tensor.data_type());
        }
    } catch (...) {
        delete[] dmtensor->dl_tensor.shape;
        delete[] dmtensor->dl_tensor.strides;
        delete dmtensor;
        throw;
    }

    py::capsule cap(dmtensor, "dltensor", [](PyObject *ptr) {
        if (PyCapsule_IsValid(ptr, "dltensor")) {
            // If consumer didn't delete the tensor,
            if (auto *dlTensor = static_cast<DLManagedTensor *>(
                    PyCapsule_GetPointer(ptr, "dltensor"))) {
                // Delete the tensor.
                if (dlTensor->deleter != nullptr) {
                    dlTensor->deleter(dlTensor);
                }
            }
        }
    });
    return cap;
}

// TODO: check what layout elayout will be -- it's from dlpack 
RocalTensor PyRocalTensor::from_dlpack(py::object src, RocalTensorLayout elayout) {
    RocalTensor rocal_tensor;
    if (hasattr(src, "__dlpack__")) {
        // Quickly check if we support the device
        if (hasattr(src, "__dlpack_device__")) {
            py::tuple dlpackDevice =
                src.attr("__dlpack_device__")().cast<py::tuple>();
            auto devType =
                static_cast<DLDeviceType>(dlpackDevice[0].cast<int>());
            if (!supported_dl_device_type(devType)) {
                throw std::runtime_error(
                    "Tensor device type is not supported.");
            }
        }

        py::capsule cap = src.attr("__dlpack__")().cast<py::capsule>();

        if (DLManagedTensor *tensor =
                static_cast<DLManagedTensor *>(cap.get_pointer())) {
            // set up
            cap.set_name("used_dltensor");
            DLTensor dtensor = tensor->dl_tensor;

            //
            // do work
            //

            // device
            if (!supported_dl_device_type(dtensor.device.device_type)) {
                throw std::runtime_error(
                    "Tensor device type is not supported.");
            }
            RocalOutputMemType dloc = get_data_location(dtensor.device.device_type);

            // dtype
            RocalTensorOutputType dtype(get_data_type(dtensor.dtype));

            // layout
            //todo: change line below according to dlpack layout type
            RocalTensorLayout layout(elayout);
            rocal_tensor->set_tensor_layout(layout);

            // ndim
            unsigned ndim = rocal_tensor->num_of_dims();
            ndim = dtensor.ndim == 0 ? 1 : static_cast<unsigned>(dtensor.ndim);
            if (ndim < 1 || ndim > TENSOR_MAX_RANK) {
                throw std::runtime_error("Tensor ndim is invalid.");
            }

            // shape 
            if (!dtensor.shape) {
                throw std::runtime_error("Tensor shape is null.");
            }
            std::vector<size_t> dims;
            for (int i = 0; i < rocal_tensor->num_of_dims(); i++) {
                dims[i] = static_cast<size_t>(dtensor.shape[i]);
            }
            // calculates strides within this function
            rocal_tensor->set_dims(dims);

            // clean up - dlpack
            if (tensor->deleter) {
                tensor->deleter(tensor);
            }
        }

        return rocal_tensor;
    }

    throw std::runtime_error("Object does not contain a __dlpack__ attribute.");
}

void PyRocalTensor::ExportUtil(py::module &m) {
    // rocalTensor for dlpack binding
    py::class_<PyRocalTensor>(m, "pyRocalTensor")
        .def("__dlpack__", &PyRocalTensor::dlpack,
            py::return_value_policy::reference,
            R"code(Export the tensor as capsule that contains a DLPackManagedTensor.)code")
        .def("__dlpack_device__", &PyRocalTensor::dlpack_device,
            py::return_value_policy::reference,
            R"code(Generate a tuple containing device info of the tensor.)code")
        .def("fromDlpack", &PyRocalTensor::from_dlpack,
            py::return_value_policy::reference,
            R"code(Wrap an existing object into a Rocal Tensor.)code");
}