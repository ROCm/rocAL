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
#include "pipeline/pipeline_operator.h"

class PipelineSerializer {
public:
    PipelineSerializer() {}
    ~PipelineSerializer() {}

    // Serialization methods
    /**
     * @brief Serialize a rocAL pipeline to a file
     * @param context The rocAL context containing the pipeline
     * @param file_path Path to save the serialized pipeline
     * @return RocalStatus indicating success or failure
     */
    void serialize_to_file(const std::string& file_path);

    /**
     * @brief Serialize a rocAL pipeline to a string
     * @param context The rocAL context containing the pipeline
     * @param serialized_string Output string containing the serialized pipeline
     * @return RocalStatus indicating success or failure
     */
    void serialize_to_string(std::string& serialized_string);

    void serialize_pipeline_config(size_t num_threads, size_t batch_size, int device_id, RocalMemType device_type, size_t prefetch_queue_depth);
    void serialize_output_tensors(TensorList& output_tensors_list);
    void serialize_operators(std::vector<std::shared_ptr<PipelineOperator>>& operators);

    /**
     * @brief Deserialize a rocAL pipeline from a file
     * @param file_path Path to the serialized pipeline file
     * @param context Output context containing the deserialized pipeline
     * @return RocalStatus indicating success or failure
     */
    // RocalStatus deserialize_from_file(const std::string& file_path, Context** context);

    /**
     * @brief Deserialize a rocAL pipeline from a string
     * @param serialized_string String containing the serialized pipeline
     * @param context Output context containing the deserialized pipeline
     * @return RocalStatus indicating success or failure
     */
    // RocalStatus deserialize_from_string(const std::string& serialized_string, Context** context);

protected:
    rocal_proto::PipelineDef _pipeline;

};
