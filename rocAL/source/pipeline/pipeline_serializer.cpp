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

#include "pipeline/pipeline_serializer.h"

void PipelineSerializer::serialize_to_string(std::string& serialized_string) {
    serialized_string = _pipeline.SerializeAsString();
}

void PipelineSerializer::serialize_pipeline_config(size_t num_threads, size_t batch_size, int device_id, RocalMemType device_type, size_t prefetch_queue_depth) {
    _pipeline.set_num_threads(num_threads);
    _pipeline.set_batch_size(batch_size);
    _pipeline.set_device_id(device_id);
    _pipeline.set_rocal_cpu(device_type == RocalMemType::HOST ? true : false);
    _pipeline.set_prefetch_queue_depth(prefetch_queue_depth);
}

void PipelineSerializer::serialize_operators(std::vector<std::shared_ptr<PipelineOperator>>& operators) {
    // Serialize all operators
    for (auto &pipe_op : operators) {
        rocal_proto::OperatorDef *op = _pipeline.add_operators();
        op->set_name(pipe_op->operator_name);
        op->set_module_name(pipe_op->module_name);
        // Add support to add each argument in the operator
        pipe_op->serialize_pipeop_args_to_protobuf(op);
        pipe_op->serialize_pipeop_inputs_and_outputs_to_protobuf(op);
    }
}

void PipelineSerializer::serialize_output_tensors(TensorList& output_tensors_list) {

    // Serialize the pipeline outputs
    for (size_t idx = 0; idx < output_tensors_list.size(); idx++) {
        rocal_proto::InputOutput *output = _pipeline.add_pipe_outputs();
        auto pipe_output = output_tensors_list[idx];
        output->set_name(pipe_output->tensor_name());
        output->set_device(static_cast<int>(pipe_output->info().mem_type()));
        output->set_dtype(static_cast<int>(pipe_output->info().data_type()));
        output->set_layout(static_cast<int>(pipe_output->info().layout()));
        output->set_color_format(static_cast<int>(pipe_output->info().color_format()));
        for (auto& dim : pipe_output->info().dims())
            output->add_dims(dim);
        output->set_num_dims(pipe_output->info().num_of_dims());
        output->set_is_argument_input(false);
    }
}
