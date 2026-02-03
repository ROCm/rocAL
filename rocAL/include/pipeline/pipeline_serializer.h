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

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "pipeline/pipeline_operator.h"
#include "rocal.pb.h"

/**
 * @brief Helper to serialize a built rocAL pipeline into protobuffers.
 *
 * This class gathers top-level pipeline configuration, operators (names, modules,
 * arguments) and tensors (inputs/outputs) and writes them into rocal_proto::PipelineDef.
 */
class PipelineSerializer {
public:
    PipelineSerializer() {}
    ~PipelineSerializer() {}

    // Serialization methods
    /**
     * @brief Serialize the current PipelineDef into a binary string.
     * @param serialized_string Output string containing the serialized pipeline
     */
    void serialize_to_string(std::string& serialized_string);

    /**
     * @brief Serialize global pipeline configuration.
     */
    void serialize_pipeline_config(size_t num_threads, size_t batch_size, int device_id, RocalMemType device_type, size_t prefetch_queue_depth, size_t seed);
    /**
     * @brief Serialize pipeline output tensors (shape, dtype, device, layout).
     */
    void serialize_output_tensors(TensorList& output_tensors_list);
    /**
     * @brief Serialize all operators in the pipeline, their arguments, and IO tensors.
     */
    void serialize_operators(std::vector<std::shared_ptr<PipelineOperator>>& operators);
    /**
     * @brief Serialize a single operator's arguments into protobuf.
     */
    void serialize_pipeop_arguments(const ArgumentSet& arguments_list, rocal_proto::OperatorDef *opdef);

    /**
     * @brief Deserialize operator arguments from protobuf into Argument objects.
     */
    RocalStatus deserialize_args_from_protobuf(const rocal_proto::OperatorDef& opdef, ArgumentSet& arguments);

    /**
     * @brief Clear any previously serialized state to start fresh.
     */
    void reset();

protected:
    rocal_proto::PipelineDef _pipeline_proto;

};
