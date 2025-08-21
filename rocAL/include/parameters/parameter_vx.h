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

#pragma once
#include <VX/vx_compatibility.h>
#include <VX/vx_types.h>

#include <vector>
#include "pipeline/tensor.h"
#include "parameters/parameter_factory.h"

template <typename T>
class ParameterVX {
   public:
    ParameterVX(unsigned ovx_param_idx, T default_range_start, T default_range_end) : OVX_PARAM_IDX(ovx_param_idx),
                                                                                      _DEFAULT_RANGE_START(default_range_start),
                                                                                      _DEFAULT_RANGE_END(default_range_end) {
        _param = ParameterFactory::instance()->create_uniform_rand_param<T>(_DEFAULT_RANGE_START,
                                                                            _DEFAULT_RANGE_END);
    }
    ParameterVX(T default_range_start, T default_range_end) : _DEFAULT_RANGE_START(default_range_start),
                                                              _DEFAULT_RANGE_END(default_range_end) {
        _param = ParameterFactory::instance()->create_uniform_rand_param<T>(_DEFAULT_RANGE_START,
                                                                            _DEFAULT_RANGE_END);
    }
    void create(vx_node node) {
        vx_status status;
        auto ref = vxGetParameterByIndex(node, OVX_PARAM_IDX);
        if ((status = vxQueryParameter(ref, VX_PARAMETER_ATTRIBUTE_REF, &_scalar, sizeof(vx_scalar))) != VX_SUCCESS ||
            (status = vxGetStatus((vx_reference)node)) != VX_SUCCESS)
            THROW("Getting vx scalar from the vx node failed" + TOSTR(status));
        if ((status = vxReadScalarValue(_scalar, &_val)) != VX_SUCCESS)
            THROW("Reading vx scalar failed" + TOSTR(status));
    }
    void create_array(std::shared_ptr<Graph> graph, vx_enum data_type, unsigned batch_size) {
        _batch_size = batch_size;
        _arrVal.resize(_batch_size);
        _array = vxCreateArray(vxGetContext((vx_reference)graph->get()), data_type, _batch_size);
        vxAddArrayItems(_array, _batch_size, _arrVal.data(), sizeof(T));
        update_array();
    }
    void create_tensor(std::shared_ptr<Graph> graph, TensorInfo info) {
       std::cerr << "\n CREATE TENSOR with info";
        vx_enum tensor_data_type = interpret_tensor_data_type(info.data_type()); // TODO: Add this function for interpretation
        _tensor = vxCreateTensor(vxGetContext((vx_reference)graph->get()), info.num_of_dims(), info.dims().data(), tensor_data_type, 0);

    }


    void create_tensor(std::shared_ptr<Graph> graph, vx_enum data_type, unsigned batch_size) {
        std::cerr << "Comes here - vxCreateTensorFromHandle";
        std::cerr << "\n CREATE TENSOR with dtype and batch_size";
        vx_size dims[1] = { batch_size };
        _batch_size = batch_size;
        vx_size output_dims[1];
        size_t num_tensor_dims;
        _arrVal.resize(batch_size);
        vx_size stride_output[1] = {sizeof(_arrVal[0])};
        std::cerr << "stride_output :: " << stride_output[0] << "\t sizeof(_arrVal) : " << sizeof(_arrVal) << "\n sizeof(_arrVal[0]) : " << sizeof(_arrVal[0]);
        _tensor = vxCreateTensorFromHandle(vxGetContext((vx_reference)graph->get()), 1, dims, data_type, 0, stride_output, _arrVal.data(), VX_MEMORY_TYPE_HOST);
        update_tensor();
    }
    void set_tensor(vx_tensor external_source_tensor) {
        _tensor = external_source_tensor;  
        if(_tensor == nullptr)
            std::cerr << "\n Tensor is a null PTR"; 
    }
    
    void set_param(Parameter<T>* param) {
        if (!param)
            return;

        ParameterFactory::instance()->destroy_param(_param);
        _param = param;
    }
    void set_param(T val) {
        ParameterFactory::instance()->destroy_param(_param);
        _param = ParameterFactory::instance()->create_single_value_param(val);
    }
    T default_value() {
        return _param->default_value();
    }
    vx_array default_array() {
        return _array;
    }
    vx_tensor default_tensor() {
        return _tensor;
    }
    vx_scalar default_scalar(std::shared_ptr<Graph> _graph, vx_enum data_type) {
        _scalar = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), data_type, &_val);
        return _scalar;
    }
    T get() {
        return _val;
    }
    void update() {
        vx_status status;

        T val = _param->get();

        if (_val == val)
            return;

        if ((status = vxWriteScalarValue(_scalar, &val)) != VX_SUCCESS)
            WRN("Updating vx scalar failed")
    }
    void update_array() {
        vx_status status;
        for (uint i = 0; i < _batch_size; i++) {
            _arrVal[i] = renew();
            std::cerr << "\n _arrVal[i] : " << _arrVal[i];
            // INFO("update_array: " + TOSTR(i) + "," + TOSTR(_arrVal[i]));
        }
        status = vxCopyArrayRange((vx_array)_array, 0, _batch_size, sizeof(T), _arrVal.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != 0)
            THROW(" vxCopyArrayRange failed in update_array (ParameterVX): " + TOSTR(status))
    }
    
    void update_tensor(TensorInfo info) {
        vx_status status;
        for (uint i = 0; i < _batch_size; i++) { 
            _arrVal[i] = renew();
            std::cerr << "\n _arrVal[i] : " << _arrVal[i];
        }
        status = vxCopyTensorPatch((vx_tensor)_tensor, info.num_of_dims(), nullptr, nullptr, info.strides(), _arrVal.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != VX_SUCCESS)
            THROW("ERROR: vxCopyTensorPatch: write failed" + TOSTR(status));
    }

    void update_tensor() {
        vx_status status;
        for (uint i = 0; i < _batch_size; i++) { 
            _arrVal[i] = renew();
            std::cerr << "\n _arrVal[i] : " << _arrVal[i];
        }
        vx_size stride_output[1] = {sizeof(float)};
        vx_size output_dims[1];
    }
    
    //! Converts the Rocal data_type to OpenVX
vx_enum interpret_tensor_data_type(RocalTensorDataType data_type) {
    switch (data_type) {
        case RocalTensorDataType::FP32:
            return VX_TYPE_FLOAT32;
        case RocalTensorDataType::FP16:
            return VX_TYPE_FLOAT16;
        case RocalTensorDataType::UINT8:
            return VX_TYPE_UINT8;
        case RocalTensorDataType::INT8:
            return VX_TYPE_INT8;
        case RocalTensorDataType::INT32:
            return VX_TYPE_INT32;
        case RocalTensorDataType::UINT32:
            return VX_TYPE_UINT32;
        default:
            THROW("Unsupported Tensor type " + TOSTR(data_type))
    }
}
    
    T renew() {
        _param->renew();
        return _param->get();
    }

    std::vector<T> get_array() {
        return _param->get_array();
    }

    ~ParameterVX() {
        if(_tensor)
            vxReleaseTensor((vx_tensor*)&_tensor);
    }

   private:
    vx_scalar _scalar;
    vx_array _array = nullptr;
    vx_tensor _tensor = nullptr;
    Parameter<T>* _param;
    T _val;
    std::vector<T> _arrVal;
    unsigned _batch_size;
    unsigned OVX_PARAM_IDX;
    const T _DEFAULT_RANGE_START;
    const T _DEFAULT_RANGE_END;
};
