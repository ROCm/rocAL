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
#include <memory>
#include <unordered_map>
#include <functional>
#include <string>
#include <vector>
#include "pipeline/commons.h"

// Forward declarations
class Node;
class Tensor;

/*!
 * \brief Factory class for dynamic node creation and registration
 * 
 * This singleton class implements the factory pattern for managing node registration and creation,
 * which is essential for deserializing pipeline graphs. It maintains separate registries for
 * loader nodes and augmentation nodes, allowing runtime instantiation of nodes by name.
 * 
 * The factory supports two types of nodes:
 * - Loader nodes: Created with a Tensor output and device resources
 * - Augmentation nodes: Created with input and output Tensor vectors
 * 
 * Nodes are registered using the REGISTER_LOADER_NODE and REGISTER_NODE macros.
 */
class NodeFactory {
public:
    using LoaderCreator = std::function<std::shared_ptr<Node>(Tensor*, void*)>;
    using AugmentationCreator = std::function<std::shared_ptr<Node>(const std::vector<Tensor *>&, const std::vector<Tensor *>&)>;

    static NodeFactory& instance() {
        static NodeFactory factory;
        return factory;
    }

    void register_loader_node(const std::string& name, LoaderCreator creator) {
        _loader_node_registry[name] = std::move(creator);
    }

    void register_node(const std::string& name, AugmentationCreator creator) {
        _node_registry[name] = std::move(creator);
    }

    std::shared_ptr<Node> create_loader_node(const std::string& name, Tensor* output_tensor, void *dev_resource) const {
        auto it = _loader_node_registry.find(name);
        if (it != _loader_node_registry.end()) {
            return it->second(output_tensor, dev_resource);
        } else {
            THROW("LoaderNode not found in the registry: " + name);
        }
    }

    std::shared_ptr<Node> create_node(const std::string& name, const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs) const {
        auto it = _node_registry.find(name);
        if (it != _node_registry.end()) {
            return it->second(inputs, outputs);
        } else {
            THROW("Node not found in the registry: " + name);
        }
    }

private:
    // Private constructor to enforce singleton pattern
    NodeFactory() = default;
    
    std::unordered_map<std::string, LoaderCreator> _loader_node_registry;
    std::unordered_map<std::string, AugmentationCreator> _node_registry;

    // Delete copy constructor and copy-assignment operator to prevent copying
    NodeFactory(const NodeFactory&) = delete;
    NodeFactory& operator=(const NodeFactory&) = delete;
};

/*!
 * \brief Macro for automatic loader node registration
 * \param CLASS_NAME The loader node class to register
 * 
 * This macro automatically registers a loader node class with the NodeFactory.
 * 
 * Usage: REGISTER_LOADER_NODE(ImageLoaderNode)
 * 
 * The registration happens at static initialization time, ensuring the node
 * is available for deserialization before main() executes.
 */
#define REGISTER_LOADER_NODE(CLASS_NAME) \
    static struct CLASS_NAME##_NodeRegistrar { \
        CLASS_NAME##_NodeRegistrar() { \
            NodeFactory::instance().register_loader_node(#CLASS_NAME, [](Tensor *output, void *dev_resources) { \
                return std::make_shared<CLASS_NAME>(output, dev_resources); \
            }); \
        } \
    } _##CLASS_NAME##_registrar;

/*!
 * \brief Macro for automatic augmentation node registration
 * \param CLASS_NAME The augmentation node class to register
 * 
 * This macro automatically registers an augmentation node class with the NodeFactory.
 * 
 * Usage: REGISTER_NODE(BrightnessNode)
 * 
 * The registration happens at static initialization time, ensuring the node
 * is available for deserialization before main() executes.
 */
#define REGISTER_NODE(CLASS_NAME) \
    static struct CLASS_NAME##_NodeRegistrar { \
        CLASS_NAME##_NodeRegistrar() { \
            NodeFactory::instance().register_node(#CLASS_NAME, [](const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs) { \
                return std::make_shared<CLASS_NAME>(inputs, outputs); \
            }); \
        } \
    } _##CLASS_NAME##_registrar;
