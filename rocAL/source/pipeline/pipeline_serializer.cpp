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

#include <fstream>

#include "parameters/parameter_factory.h"

void PipelineSerializer::serialize_to_string(std::string& serialized_string) {
    if (!_pipeline_proto.SerializeToString(&serialized_string)) {
        THROW("Failed to serialize pipeline to string.");
    }
}

void PipelineSerializer::serialize_pipeline_config(size_t num_threads, size_t batch_size, int device_id, RocalMemType device_type, size_t prefetch_queue_depth, size_t seed) {
    _pipeline_proto.set_num_threads(static_cast<uint64_t>(num_threads));
    _pipeline_proto.set_batch_size(static_cast<uint64_t>(batch_size));
    _pipeline_proto.set_device_id(device_id);
    _pipeline_proto.set_rocal_cpu(device_type == RocalMemType::HOST);
    _pipeline_proto.set_prefetch_queue_depth(static_cast<uint64_t>(prefetch_queue_depth));
    _pipeline_proto.set_seed(static_cast<uint64_t>(seed));
}

void set_tensor_proto(rocal_proto::InputOutput *in_out_proto, Tensor *tensor, bool is_input) {
    in_out_proto->set_name(tensor->tensor_name());
    in_out_proto->set_device(static_cast<int>(tensor->info().mem_type()));
    in_out_proto->set_dtype(static_cast<int>(tensor->info().data_type()));
    in_out_proto->set_layout(static_cast<int>(tensor->info().layout()));
    in_out_proto->set_color_format(static_cast<int>(tensor->info().color_format()));
    for (auto &dim : tensor->info().dims())
        in_out_proto->add_dims(dim);
    in_out_proto->set_num_dims(tensor->info().num_of_dims());
    in_out_proto->set_is_argument_input(is_input);
}

// Template helper function to add parameter values based on type
template<typename T>
void add_param_value(rocal_proto::Parameter *parameter, const T& value) {
    if constexpr (std::is_same_v<T, int>) {
        parameter->add_param_val_int(value);
    } else if constexpr (std::is_same_v<T, float>) {
        parameter->add_param_val_float(value);
    }
}

// Template function to safely extract and cast parameter core
template<typename T>
auto extract_param_core(const Argument &op_arg) {
    if constexpr (std::is_same_v<T, int>) {
        auto param = std::get<IntParam *>(op_arg.param);
        return param->core;
    } else if constexpr (std::is_same_v<T, float>) {
        auto param = std::get<FloatParam *>(op_arg.param);
        return param->core;
    } else {
        THROW("Extract_param_core only supports int and float types.")
    }
}

// Template function to handle SimpleParameter serialization
template<typename T>
void serialize_simple_parameter(rocal_proto::Parameter *parameter, const Argument &op_arg) {
    auto param_core = extract_param_core<T>(op_arg);
    auto simple_param = dynamic_cast<SimpleParameter<T> *>(param_core);
    if (!simple_param) {
        THROW("Failed to cast parameter '" + op_arg.arg_name + "' to SimpleParameter type.");
    }
    add_param_value(parameter, simple_param->get());
}

// Template function to handle UniformRand serialization
template<typename T>
void serialize_uniform_rand(rocal_proto::Parameter *parameter, const Argument &op_arg) {
    auto param_core = extract_param_core<T>(op_arg);
    auto uniform_param = dynamic_cast<UniformRand<T> *>(param_core);
    if (!uniform_param) {
        THROW("Failed to cast parameter '" + op_arg.arg_name + "' to UniformRand type.");
    }
    auto uniform_range = uniform_param->get_start_and_end();
    add_param_value(parameter, uniform_range.first);
    add_param_value(parameter, uniform_range.second);
}

// Template function to handle CustomRand serialization
template<typename T>
void serialize_custom_rand(rocal_proto::Parameter *parameter, const Argument &op_arg) {
    auto param_core = extract_param_core<T>(op_arg);
    auto random_param = dynamic_cast<CustomRand<T> *>(param_core);
    if (!random_param) {
        THROW("Failed to cast parameter '" + op_arg.arg_name + "' to CustomRand type.");
    }
    // Add values
    auto values_vec = random_param->get_values();
    for (const auto &val : values_vec) {
        add_param_value(parameter, val);
    }
    
    // Add frequencies
    auto frequency_vec = random_param->get_frequencies();
    for (const auto &val : frequency_vec) {
        parameter->add_frequency(val);
    }
    
    parameter->set_size(random_param->size());
}

// Type dispatcher function to handle different data types
template<typename T>
void serialize_parameter_by_type(rocal_proto::Parameter *parameter, const Argument &op_arg) {
    if (op_arg.sub_type_name == "SimpleParameter") {
        serialize_simple_parameter<T>(parameter, op_arg);
    } else if (op_arg.sub_type_name == "UniformRand") {
        serialize_uniform_rand<T>(parameter, op_arg);
    } else if (op_arg.sub_type_name == "CustomRand") {
        serialize_custom_rand<T>(parameter, op_arg);
    }
}

// Main function with improved structure and error handling
void serialize_parameter_to_protobuf(rocal_proto::Parameter *parameter, const Argument &op_arg) {
    if (op_arg.sub_type_name == "SimpleParameter") {
        parameter->set_param_type(static_cast<int>(RocalParameterType::DETERMINISTIC));
    } else if (op_arg.sub_type_name == "UniformRand") {
        parameter->set_param_type(static_cast<int>(RocalParameterType::RANDOM_UNIFORM));
    } else if (op_arg.sub_type_name == "CustomRand") {
        parameter->set_param_type(static_cast<int>(RocalParameterType::RANDOM_CUSTOM));
    } else {
        THROW("Unknown parameter type '" + op_arg.sub_type_name + "' for argument '" + op_arg.arg_name + "'");
    }

    if (op_arg.type_name == "int") {
        serialize_parameter_by_type<int>(parameter, op_arg);
    } else if (op_arg.type_name == "float") {
        serialize_parameter_by_type<float>(parameter, op_arg);
    }
}

void PipelineSerializer::serialize_pipeop_arguments(const ArgumentSet& arguments_list, rocal_proto::OperatorDef *opdef) {

    // Iterate through each argument to store in the protobuffers
    for (auto &arg_pair : arguments_list) {
        const Argument& op_arg = arg_pair.second;
        rocal_proto::Arguments *arg = opdef->add_args();
        arg->set_name(op_arg.arg_name);
        arg->set_type(op_arg.type_name);
        arg->set_is_vector(op_arg.is_vector);

        if (op_arg.type_name == "nullptr") continue;
        if (op_arg.sub_type_name != "")
            arg->set_instance_name(op_arg.sub_type_name);

        if (op_arg.is_parameter) {
            rocal_proto::Parameter *param = arg->mutable_param();
            serialize_parameter_to_protobuf(param, op_arg);
        } else if (op_arg.type_name == "enum") {
            if (op_arg.values.empty()) {
                THROW("Enum argument '" + op_arg.arg_name + "' requires at least one value.");
            }
            rocal_proto::EnumType* enum_arg = arg->mutable_enum_value();
            enum_arg->set_name(op_arg.sub_type_name);
            enum_arg->set_value(std::any_cast<int>(op_arg.values[0]));
        } else {
            // Scalars go to the flat repeated fields; vectors go to repeated *Vector messages
            if (op_arg.is_vector) {
                if (op_arg.values.empty()) {
                    // Represent empty vector by adding an empty vector message of the right type
                    if (op_arg.type_name == "int" || op_arg.type_name == "shared_ptr"
                        || op_arg.type_name == "unsigned" || op_arg.type_name == "size_t") {
                        static_cast<void>(arg->add_int_vectors());
                    } else if (op_arg.type_name == "float") {
                        static_cast<void>(arg->add_float_vectors());
                    } else if (op_arg.type_name == "char_str" || op_arg.type_name == "string"
                               || op_arg.type_name == "map_string") {
                        static_cast<void>(arg->add_string_vectors());
                    } else {
                        THROW("Vector type not supported for Argument " + op_arg.arg_name + " with type " + op_arg.type_name + ".");
                    }
                } else {
                    if (op_arg.type_name == "int" || op_arg.type_name == "unsigned" || op_arg.type_name == "size_t") {
                        auto *vec = arg->add_int_vectors();
                        for (auto &v : op_arg.values) {
                            // Map unsigned/size_t to int64 for IntVector as per spec (only Int/Float/String vectors permitted)
                            if (op_arg.type_name == "unsigned") {
                                vec->add_values(static_cast<int64_t>(std::any_cast<unsigned>(v)));
                            } else if (op_arg.type_name == "size_t") {
                                vec->add_values(static_cast<int64_t>(std::any_cast<size_t>(v)));
                            } else {
                                vec->add_values(static_cast<int64_t>(std::any_cast<int>(v)));
                            }
                        }
                    } else if (op_arg.type_name == "float") {
                        auto *vec = arg->add_float_vectors();
                        for (auto &v : op_arg.values) {
                            vec->add_values(std::any_cast<float>(v));
                        }
                    } else if (op_arg.type_name == "char_str" || op_arg.type_name == "string" 
                               || op_arg.type_name == "map_string") {
                        auto *vec = arg->add_string_vectors();
                        for (auto &v : op_arg.values) {
                            vec->add_values(std::any_cast<std::string>(v));
                        }
                    } else {
                        THROW("Vector type not supported for Argument " + op_arg.arg_name + " with type " + op_arg.type_name + ".");
                    }
                }
            } else {
                // Scalar path (use flat repeated fields)
                if (op_arg.values.size() > 1) {
                    THROW("Argument '" + op_arg.arg_name + "' has more than one value, but is_vector is false. This is not allowed.");
                }
                for (auto &v : op_arg.values) {
                    if (op_arg.type_name == "int" || op_arg.type_name == "shared_ptr") {
                        arg->add_ints(std::any_cast<int>(v));
                    } else if (op_arg.type_name == "float") {
                        arg->add_floats(std::any_cast<float>(v));
                    } else if (op_arg.type_name == "char_str" || op_arg.type_name == "string") {
                        arg->add_strings(std::any_cast<std::string>(v));
                    } else if (op_arg.type_name == "bool") {
                        arg->add_bools(std::any_cast<bool>(v));
                    } else if (op_arg.type_name == "unsigned") {
                        arg->add_uints(std::any_cast<unsigned>(v));
                    } else if (op_arg.type_name == "size_t") {
                        arg->add_uints(std::any_cast<size_t>(v));
                    } else {
                        THROW("Invalid type specified for the Argument " + op_arg.arg_name + ".");
                    }
                }
            }
        }
    }
}

void PipelineSerializer::serialize_operators(std::vector<std::shared_ptr<PipelineOperator>>& operators) {
    // Serialize all operators
    for (auto &pipe_op : operators) {
        rocal_proto::OperatorDef *op = _pipeline_proto.add_operators();
        op->set_name(pipe_op->operator_name);
        op->set_module_name(pipe_op->module_name);
        serialize_pipeop_arguments(pipe_op->get_arguments(), op);

        if (pipe_op->module_name == "reader")
            continue;  // Readers do not have tensor outputs, hence return

        // Serialize input tensors to protobuffers
        for (auto &node_input : pipe_op->get_inputs()) {
            rocal_proto::InputOutput *input = op->add_inputs();
            set_tensor_proto(input, node_input, true);
        }

        // Serialize output tensors to protobuffers
        for (auto &node_output : pipe_op->get_outputs()) {
            rocal_proto::InputOutput *output = op->add_outputs();
            set_tensor_proto(output, node_output, false);
        }
    }
}

void PipelineSerializer::serialize_output_tensors(TensorList& output_tensors_list) {

    // Serialize the pipeline outputs
    for (size_t idx = 0; idx < output_tensors_list.size(); idx++) {
        rocal_proto::InputOutput *output = _pipeline_proto.add_pipe_outputs();
        auto pipe_output = output_tensors_list[idx];
        set_tensor_proto(output, pipe_output, false);
    }
}

RocalStatus PipelineSerializer::deserialize_args_from_protobuf(const rocal_proto::OperatorDef& opdef, ArgumentSet& arguments) {
    arguments.clear();
    for (const auto& proto_arg : opdef.args()) {
        Argument arg;
        arg.arg_name = proto_arg.name();
        arg.type_name = proto_arg.has_type() ? proto_arg.type() : "";
        arg.sub_type_name = proto_arg.has_instance_name() ? proto_arg.instance_name() : "";
        arg.is_vector = proto_arg.is_vector();
        arg.is_parameter = proto_arg.has_param();

        // Handle parameters
        if (arg.type_name == "enum") {
            const auto& enum_val = proto_arg.enum_value();
            arg.sub_type_name = enum_val.name();
            if (EnumRegistry::getInstance().isEnumRegistered(arg.sub_type_name)) {
                // Use new std::any-based approach
                std::any enum_value = EnumRegistry::getInstance().convertIntToEnum(arg.sub_type_name, enum_val.value());
                arg.values.emplace_back(enum_value);
            } else {
                THROW("Enum type '" + arg.sub_type_name + "' is not registered. Please ensure the enum is properly registered with EnumRegistry before deserialization.");
            }
        } else if (arg.is_parameter) {
            const auto& param = proto_arg.param();
            if (arg.type_name == "int") {
                if (arg.sub_type_name == "SimpleParameter") {
                    if (param.param_val_int_size() < 1) {
                        THROW("Invalid parameter: missing value for int SimpleParameter '" + arg.arg_name + "'");
                    }
                    arg.param = static_cast<IntParam*>(ParameterFactory::instance()->create_single_value_int_param(param.param_val_int(0)));
                } else if (arg.sub_type_name == "UniformRand") {
                    if (param.param_val_int_size() < 2) {
                        THROW("Invalid parameter: missing values for UniformRand parameter " + arg.arg_name);
                    }
                    arg.param = static_cast<IntParam*>(ParameterFactory::instance()->create_uniform_int_rand_param(param.param_val_int(0), param.param_val_int(1)));
                } else if (arg.sub_type_name == "CustomRand") {
                    std::vector<int> values(param.param_val_int().begin(), param.param_val_int().end());
                    std::vector<double> freqs(param.frequency().begin(), param.frequency().end());
                    arg.param = static_cast<IntParam*>(ParameterFactory::instance()->create_custom_int_rand_param(values.data(),
                                                                      freqs.data(),
                                                                      values.size()));
                } else {
                    THROW("Invalid parameter sub-type '" + arg.sub_type_name + "' for int parameter '" + arg.arg_name + "'. Expected: SimpleParameter, UniformRand, or CustomRand");
                }
            } else if (arg.type_name == "float") {
                if (arg.sub_type_name == "SimpleParameter") {
                    if (param.param_val_float_size() < 1) {
                        THROW("Invalid parameter: missing value for float SimpleParameter '" + arg.arg_name + "'");
                    }
                    arg.param = static_cast<FloatParam*>(ParameterFactory::instance()->create_single_value_float_param(param.param_val_float(0)));
                } else if (arg.sub_type_name == "UniformRand") {
                    if (param.param_val_float_size() < 2) {
                        THROW("Invalid parameter: missing values for UniformRand parameter " + arg.arg_name);
                    }
                    arg.param = static_cast<FloatParam*>(ParameterFactory::instance()->create_uniform_float_rand_param(param.param_val_float(0), param.param_val_float(1)));
                } else if (arg.sub_type_name == "CustomRand") {
                    std::vector<float> values(param.param_val_float().begin(), param.param_val_float().end());
                    std::vector<double> freqs(param.frequency().begin(), param.frequency().end());
                    arg.param = static_cast<FloatParam*>(ParameterFactory::instance()->create_custom_float_rand_param(values.data(),
                                                                      freqs.data(),
                                                                      values.size()));
                } else {
                    THROW("Invalid parameter sub-type '" + arg.sub_type_name + "' for float parameter '" + arg.arg_name + "'. Expected: SimpleParameter, UniformRand, or CustomRand");
                }
            } else {
                // For unsupported parameter types, set as null pointer
                arg.is_null_ptr = true;
            }
        } else if (arg.is_vector) {
            // Handle vector deserialization based on type
            if (arg.type_name == "int" || arg.type_name == "unsigned" || arg.type_name == "size_t" || arg.type_name == "shared_ptr") {
                // Deserialize integer vectors - expect exactly one vector
                if (proto_arg.int_vectors_size() > 1) {
                    THROW("Expected at most one int vector for argument " + arg.arg_name + ", but found " + std::to_string(proto_arg.int_vectors_size()));
                }
                const auto& int_vec = proto_arg.int_vectors(0);
                for (auto val : int_vec.values()) {
                    if (arg.type_name == "unsigned") {
                        arg.values.emplace_back(static_cast<unsigned>(val));
                    } else if (arg.type_name == "size_t") {
                        arg.values.emplace_back(static_cast<size_t>(val));
                    } else if (arg.type_name == "shared_ptr" || arg.type_name == "int") {
                        arg.values.emplace_back(static_cast<int>(val));
                    }
                }
            } else if (arg.type_name == "float") {
                // Deserialize float vectors - expect exactly one vector
                if (proto_arg.float_vectors_size() > 1) {
                    THROW("Expected at most one float vector for argument " + arg.arg_name + ", but found " + std::to_string(proto_arg.float_vectors_size()));
                }
                const auto& float_vec = proto_arg.float_vectors(0);
                for (auto val : float_vec.values()) {
                    arg.values.emplace_back(val);
                }
            } else if (arg.type_name == "char_str" || arg.type_name == "string" || arg.type_name == "map_string") {
                // Deserialize string vectors - expect exactly one vector
                if (proto_arg.string_vectors_size() > 1) {
                    THROW("Expected at most one string vector for argument " + arg.arg_name + ", but found " + std::to_string(proto_arg.string_vectors_size()));
                }
                const auto& string_vec = proto_arg.string_vectors(0);
                for (const auto& val : string_vec.values()) {
                    arg.values.emplace_back(val);
                }
            } else {
                THROW("Vector type not supported during deserialization for Argument " + arg.arg_name + " with type " + arg.type_name);
            }
        }

        // Handle non-parameter arguments
        else if (arg.type_name == "int" || arg.type_name == "shared_ptr") {
            for (auto i : proto_arg.ints()) {
                arg.values.emplace_back(static_cast<int>(i));
            }
        } else if (arg.type_name == "float") {
            for (auto f : proto_arg.floats()) {
                arg.values.emplace_back(f);
            }
        } else if (arg.type_name == "char_str" || arg.type_name == "string") {
            for (const auto& s : proto_arg.strings()) {
                arg.values.emplace_back(s);
            }
        } else if (arg.type_name == "bool") {
            for (auto b : proto_arg.bools()) {
                arg.values.emplace_back(b);
            }
        } else if (arg.type_name == "unsigned") {
            for (auto u : proto_arg.uints()) {
                arg.values.emplace_back(static_cast<unsigned>(u));
            }
        } else if (arg.type_name == "size_t") {
            for (auto u : proto_arg.uints()) {
                arg.values.emplace_back(static_cast<size_t>(u));
            }
        } else if (arg.type_name == "nullptr") {
            arg.is_null_ptr = true;
        } else {
            THROW("Invalid or unsupported type while deserializing: " + arg.type_name);
        }

        arguments.add_argument(arg.arg_name, arg);
    }
    return ROCAL_OK;
}

void PipelineSerializer::reset() {
    _pipeline_proto.Clear();
}
