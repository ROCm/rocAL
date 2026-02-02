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
#include <memory>
#include <set>
#include <array>
#include <utility>

#include "pipeline/graph.h"
#include "loaders/loader_module.h"
#include "pipeline/tensor.h"
#include "pipeline/argument.h"
#include "pipeline/node_factory.h"

class Node {
   public:
    Node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : _inputs(inputs),
                                                                                      _outputs(outputs),
                                                                                      _batch_size(outputs[0]->info().batch_size()) {}
    virtual ~Node();
    void create(std::shared_ptr<Graph> graph);
    void update_parameters();
    const std::vector<Tensor *>& input() const { return _inputs; }
    const std::vector<Tensor *>& output() const { return _outputs; }
    void add_next(const std::shared_ptr<Node> &node);   // Adds the Node next to the current Node
    void add_previous(const std::shared_ptr<Node> &node);   // Adds the Node preceding the current Node
    void release();
    std::shared_ptr<Graph> graph() { return _graph; }
    void set_meta_data(pMetaDataBatch meta_data_info) { _meta_data_info = meta_data_info; }
    bool _is_ssd = false;
    const Roi2DCords *get_src_roi() { return _inputs[0]->info().roi().get_2D_roi(); }
    const Roi2DCords *get_dst_roi() { return _outputs[0]->info().roi().get_2D_roi(); }
    void set_graph_id(int id) { _graph_id = id; }
    int get_graph_id() { return _graph_id; }
    virtual std::string node_name() const { return ""; }
    const ArgumentSet& get_args_list() const { return _args; }
    // Returns the LoaderModule associated with this node, Derived LoaderNodes should override this method.
    virtual std::shared_ptr<LoaderModule> get_loader_module() { THROW("get_loader_module() is not implemented for the Node"); }
    // Overloaded method for loader nodes to initialize the arguments during deserialization.
    virtual void initialize_args(const ArgumentSet& arguments, std::shared_ptr<MetaDataReader> meta_data_reader) { 
        THROW("initialize_args not implemented for the LoaderNode type: " + node_name() + 
              ". Derived LoaderNode must override this method to handle deserialization with metadata reader."); 
    }
    // Overloaded method for augmentation nodes to initialize the arguments during deserialization.
    virtual void initialize_args(const ArgumentSet &arguments) { 
        THROW("initialize_args(arguments) not implemented for node type: " + node_name() + 
              ". Derived Nodes must override this method to handle deserialization.");
    }

   protected:
    virtual void create_node() = 0;
    virtual void update_node() = 0;
    const std::vector<Tensor *> _inputs;
    const std::vector<Tensor *> _outputs;
    std::shared_ptr<Graph> _graph = nullptr;
    vx_node _node = nullptr;
    size_t _batch_size;
    pMetaDataBatch _meta_data_info;
    std::vector<std::shared_ptr<Node>> _next;   // Stores the reference to a list of next Nodes
    std::vector<std::shared_ptr<Node>> _prev;   // Stores the reference to a list of previous Nodes
    int _graph_id = -1;
    ArgumentSet _args;
};
