/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <vx_ext_rpp.h>
#include "augmentations/node_external_source.h"
#include "pipeline/exception.h"

ExternalSourceNode::ExternalSourceNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs),
                                                                                                                    _source(),
                                                                                                                    _file_path(),
                                                                                                                    _dtype(){}

void ExternalSourceNode::create_node() {
    if (_node)
        return;
    vx_array charArray = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_CHAR, strlen(_source));
    vx_array filePathArray = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_CHAR, strlen(_file_path));
    vxAddArrayItems(charArray, strlen(_source), _source, sizeof(char));
    vxAddArrayItems(filePathArray, strlen(_file_path), _file_path, sizeof(char));
    char* stringCheck = new char[strlen(_source) + 1]; // +1 for null terminator

    vx_size charArraySize;
    vxQueryArray(charArray, VX_ARRAY_CAPACITY, &charArraySize, sizeof(charArraySize));
    vxCopyArrayRange(charArray, 0, charArraySize, sizeof(char), stringCheck, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    // Null-terminate the copied string
    stringCheck[charArraySize] = '\0';

    // Output the copied string
    std::cerr << "\n vx array " << stringCheck;

    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);
    _node = vxExtExternalSource(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), charArray, filePathArray, _dtype, input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the copy (vxCopyNode) node failed: " + TOSTR(status))
    std::cerr<<"\n End of node external Source";
}

void ExternalSourceNode::init(const char* source, const char* file_path, int dtype) {
    _source = new char[strlen(source)+1];
    _file_path = new char[strlen(file_path)+1];
    strcpy(_source, source);
    strcpy(_file_path, file_path);
    _source[strlen(source)] = '\0';
    _file_path[strlen(file_path)] = '\0';
    _dtype = dtype;
    std::cerr<<"\n in init source"<<_file_path;
}

void ExternalSourceNode::update_node() {
}
