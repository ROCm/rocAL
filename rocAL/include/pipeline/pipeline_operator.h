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
#include "pipeline/node.h"

/**
 * @brief Represents an operator in the pipeline for serialization purposes.
 * 
 * This class encapsulates metadata about pipeline operators including their name,
 * module category, arguments, and associated computational node. It is used to track
 * and serialize pipeline structure.
 */
class PipelineOperator {
   public:
    // Constructor to initialize the operator
    explicit PipelineOperator(const std::string& op_name, const std::string& op_module_name,
                              std::shared_ptr<Node> op_node = nullptr)
        : operator_name(op_name), module_name(op_module_name), node(std::move(op_node)) {
    }

    // Set the list of arguments associated with this operator
    void set_arguments(const std::vector<Argument>& arguments) {
        this->arguments = arguments;
    }

    std::string operator_name;              // Name of the operator (e.g., "ResizeNode")
    std::string module_name;                // Category of the operator (e.g., "loader", "augmentation" or "reader")
    std::vector<Argument> arguments;        // List of arguments/configurations for the operator
    std::shared_ptr<Node> node;             // Pointer to the associated computational graph node
};
