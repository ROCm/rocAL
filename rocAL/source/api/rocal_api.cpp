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

#include "rocal_api.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/message.h>

#include <cstring>
#include <exception>
#include <string>

#include "pipeline/commons.h"
#include "pipeline/context.h"
#include "rocal.pb.h"

RocalStatus ROCAL_API_CALL
rocalRelease(RocalContext p_context) {
    // Deleting context is required to call the destructor of all the member objects
    auto context = static_cast<Context*>(p_context);
    delete context;
    return ROCAL_OK;
}

RocalContext ROCAL_API_CALL
rocalCreate(
    size_t batch_size,
    RocalProcessMode affinity,
    int gpu_id,
    size_t cpu_thread_count,
    size_t prefetch_queue_depth,
    RocalTensorOutputType output_tensor_data_type,
    bool enable_checkpointing) {
    RocalContext context = nullptr;
    try {
        auto translate_process_mode = [](RocalProcessMode process_mode) {
            switch (process_mode) {
                case ROCAL_PROCESS_GPU:
                    return RocalAffinity::GPU;
                case ROCAL_PROCESS_CPU:
                    return RocalAffinity::CPU;
                default:
                    THROW("Unkown Rocal data type")
            }
        };
        auto translate_output_data_type = [](RocalTensorOutputType data_type) {
            switch (data_type) {
                case ROCAL_FP32:
                    return RocalTensorDataType::FP32;
                case ROCAL_FP16:
                    return RocalTensorDataType::FP16;
                case ROCAL_UINT8:
                    return RocalTensorDataType::UINT8;
                default:
                    THROW("Unkown Rocal data type")
            }
        };
        if (gpu_id < 0)
            ERR(STR("Negative GPU device ID passed to context creation. Setting GPU device ID to 0"));
        context = new Context(batch_size, translate_process_mode(affinity), std::max(gpu_id, 0), cpu_thread_count, prefetch_queue_depth, translate_output_data_type(output_tensor_data_type), enable_checkpointing);
        // Reset seed in case it's being randomized during context creation
    } catch (const std::exception& e) {
        ERR(STR("Failed to init the Rocal context, ") + STR(e.what()))
    }
    return context;
}

RocalStatus ROCAL_API_CALL
rocalRun(RocalContext p_context) {
    auto context = static_cast<Context*>(p_context);
    try {
        auto ret = context->master_graph->run();
        if (ret != MasterGraph::Status::OK)
            return ROCAL_RUNTIME_ERROR;
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

RocalStatus ROCAL_API_CALL
rocalVerify(RocalContext p_context) {
    auto context = static_cast<Context*>(p_context);
    try {
        context->master_graph->build();
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

RocalStatus ROCAL_API_CALL
rocalSerialize(RocalContext rocal_context, size_t *serialized_string_size) {
    auto context = static_cast<Context *>(rocal_context);
    try {
        if (!serialized_string_size) {
            return ROCAL_INVALID_PARAMETER_TYPE;
        }
        context->master_graph->serialize(serialized_string_size);
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

RocalContext ROCAL_API_CALL
rocalDeserialize(const char* serialized_pipeline, size_t serialized_string_size, RocalPipelineParams* pipe_params) {
    RocalContext context = nullptr;
    try {
        if (!serialized_pipeline || serialized_string_size == 0) {
            THROW("Serialized pipeline buffer is null or empty");
        }
        if (!pipe_params) {
            THROW("Invalid pipeline params pointer passed to rocalDeserialize");
        }

        // Parse from the serialized string.
        rocal_proto::PipelineDef pipe;
        google::protobuf::io::CodedInputStream coded_input(
            reinterpret_cast<const uint8_t *>(serialized_pipeline), serialized_string_size);
        coded_input.SetTotalBytesLimit(static_cast<int>(serialized_string_size));
        if (!pipe.ParseFromCodedStream(&coded_input)) {
            THROW("Failed to parse serialized pipeline protobuf");
        }

        // Get the pipeline related info
        if (!pipe_params->batch_size.has_value())
            pipe_params->batch_size = pipe.has_batch_size() ? pipe.batch_size() : 1;

        if (!pipe_params->device_id.has_value() && pipe.has_device_id())
            pipe_params->device_id = pipe.device_id();

        if (!pipe_params->num_threads.has_value() && pipe.has_num_threads())
            pipe_params->num_threads = pipe.num_threads();

        if (!pipe_params->rocal_cpu.has_value() && pipe.has_rocal_cpu())
            pipe_params->rocal_cpu = pipe.rocal_cpu();

        if (!pipe_params->prefetch_queue_depth.has_value() && pipe.has_prefetch_queue_depth())
            pipe_params->prefetch_queue_depth = pipe.prefetch_queue_depth();

        if (!pipe_params->seed.has_value() && pipe.has_seed()) {
            pipe_params->seed = pipe.seed();
            rocalSetSeed(pipe.seed());
        }

        const size_t batch_size = pipe_params->batch_size.value();
        const int device_id = pipe_params->device_id.value_or(0);
        if (device_id < 0) {
            THROW("Invalid device_id: negative values are not allowed (" + std::to_string(device_id) + ")");
        }
        const size_t num_threads = pipe_params->num_threads.value_or(1);
        const size_t prefetch_queue_depth = pipe_params->prefetch_queue_depth.value_or(3);
        const bool use_cpu = pipe_params->rocal_cpu.value_or(false);

        RocalAffinity affinity = use_cpu ? RocalAffinity::CPU : RocalAffinity::GPU;
        // Create the context
        context = new Context(batch_size, affinity,
                              device_id,
                              num_threads,
                              prefetch_queue_depth,
                              RocalTensorDataType::FP32);  // Pipeline creation currently omit dtype; defaulting to FP32.
        static_cast<Context*>(context)->master_graph->deserialize(&pipe);

    } catch (const std::exception& e) {
        delete static_cast<Context*>(context);
        context = nullptr;
        ERR(STR("Failed to init the Rocal context, ") + e.what())
    }
    return context;
}

RocalStatus ROCAL_API_CALL
rocalGetSerializedString(RocalContext rocal_context, char* serialized_string) {
    auto context = static_cast<Context*>(rocal_context);
    try {
        if (!serialized_string) {
            THROW("String copy failed, Invalid pointer passed for serialize.")
        }

        auto& serialize_pipe_string = context->master_graph->get_serialized_string();
        if (serialize_pipe_string.empty())
            THROW("Serialized string is empty, invoke rocalSerialize before obtaining the string.")
        std::memcpy(serialized_string, serialize_pipe_string.c_str(), serialize_pipe_string.size() + 1);

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

// Serialize the current pipeline state into an internal checkpoint blob.
RocalStatus ROCAL_API_CALL
rocalCheckpoint(RocalContext rocal_context, size_t *serialized_ckpt_string_size) {
    auto context = static_cast<Context*>(rocal_context);
    try {
        if (!serialized_ckpt_string_size) {
            THROW("Serialized checkpoint size pointer is null")
        }
        context->master_graph->get_serialized_checkpoint(*serialized_ckpt_string_size);
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

// Copy the last serialized checkpoint blob into the user-provided buffer.
RocalStatus ROCAL_API_CALL
rocalGetSerializedCheckpointString(RocalContext rocal_context, char* serialized_ckpt_string) {
    auto context = static_cast<Context*>(rocal_context);
    try {
        if (!serialized_ckpt_string) {
            THROW("String copy failed, Invalid pointer passed for serialize")
        }

        auto pipe_ckpt_string = context->master_graph->get_serialized_checkpoint_string();
        if (pipe_ckpt_string.empty())
            THROW("Serialized string is empty, Invoke rocalCheckpoint before obtaining the string")
        std::memcpy(serialized_ckpt_string, pipe_ckpt_string.data(), pipe_ckpt_string.size());

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}
