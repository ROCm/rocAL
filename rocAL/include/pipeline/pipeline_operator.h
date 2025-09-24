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
#include "pipeline/node.h"
#include "rocal.pb.h"

// Stores the information for the operator in the pipeline
class PipelineOperator {
   public:
    explicit inline PipelineOperator(std::string op_name, std::string op_module_name,
                                     std::shared_ptr<Node> op_node = nullptr) {
        operator_name = op_name;
        module_name = op_module_name;
        node = op_node;
    }
    void set_arguments(std::vector<Argument> op_arguments) {
        arguments = op_arguments;
    }
    void serialize_pipeop_args_to_protobuf(rocal_proto::OperatorDef *opdef);
    void serialize_pipeop_inputs_and_outputs_to_protobuf(rocal_proto::OperatorDef *opdef);
    std::string operator_name;  // Name of the Node/operator
    std::string module_name;    // Denotes the type of operator i.e loader/reader/augmentation
    std::vector<Argument> arguments;
    std::shared_ptr<Node> node;
};
