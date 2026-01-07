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

#include "loaders/image/node_cifar10_loader.h"

#include "pipeline/exception.h"

#define INIT_ARGS_COUNT 7  // Modify in accordance with number of args in init

REGISTER_LOADER_NODE(Cifar10LoaderNode)

Cifar10LoaderNode::Cifar10LoaderNode(Tensor *output, void *device_resources) : Node({}, {output}) {
    _loader_module = std::make_shared<CIFAR10Loader>(device_resources);
}

void Cifar10LoaderNode::init(const std::string &source_path, const std::string &json_path, StorageType storage_type,
                             bool loop, size_t load_batch_count, RocalMemType mem_type, const std::string &file_prefix) {
    if (!_loader_module)
        THROW("ERROR: loader module is not set for Cifar10LoaderNode, cannot initialize")
    _loader_module->set_output(_outputs[0]);
    // Set reader and decoder config accordingly for the Cifar10LoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, json_path, std::map<std::string, std::string>(), loop);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_file_prefix(file_prefix);
    // DecoderConfig will be ignored in loader. Just passing it for api match
    _loader_module->initialize(reader_cfg, DecoderConfig(DecoderType::TURBO_JPEG),
                               mem_type, _batch_size);
    _loader_module->start_loading();

    std::array<std::string, 7> arg_names = {"source_path", "json_path", "storage_type", "loop", "load_batch_count", "mem_type", "file_prefix"};
    set_node_arguments(arg_names, std::make_index_sequence<arg_names.size()>{}, source_path, json_path, storage_type, loop, load_batch_count, mem_type, file_prefix);
}

std::shared_ptr<LoaderModule> Cifar10LoaderNode::get_loader_module() {
    if (!_loader_module)
        WRN("Cifar10LoaderNode's loader module is null, not initialized")
    return _loader_module;
}

Cifar10LoaderNode::~Cifar10LoaderNode() {
    _loader_module = nullptr;
}

void Cifar10LoaderNode::initialize_args(std::vector<Argument> &arguments, std::shared_ptr<MetaDataReader> meta_data_reader) {
    (void)meta_data_reader;
    if (arguments.size() != INIT_ARGS_COUNT)
        THROW("Cifar10LoaderNode expected " + std::to_string(INIT_ARGS_COUNT) + " arguments, received " + std::to_string(arguments.size()) +
              ". Ensure all arguments present in init are accounted for");

    this->init(arguments[0].get<std::string>(), arguments[1].get<std::string>(), arguments[2].get<StorageType>(),
               arguments[3].get<bool>(), arguments[4].get<size_t>(), arguments[5].get<RocalMemType>(),
               arguments[6].get<std::string>());
}
