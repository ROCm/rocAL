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

#include "loaders/image/node_cifar10_loader_single_shard.h"

#include "pipeline/exception.h"

#define INIT_ARGS_COUNT 13  // Modify in accordance with number of args in init

REGISTER_LOADER_NODE(CIFAR10LoaderSingleShardNode)

CIFAR10LoaderSingleShardNode::CIFAR10LoaderSingleShardNode(Tensor *output, void *device_resources) : Node({}, {output}) {
    _loader_module = std::make_shared<CIFAR10LoaderSharded>(device_resources);
}

void CIFAR10LoaderSingleShardNode::init(unsigned shard_id, unsigned shard_count, const std::string &source_path, StorageType storage_type,
                                      bool shuffle, bool loop, size_t load_batch_count, RocalMemType mem_type, const std::string &file_prefix, const ShardingInfo& sharding_info) {
    if (!_loader_module)
        THROW("ERROR: loader module is not set for CIFAR10LoaderSingleShardNode, cannot initialize")
    if (shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    if (shard_id >= shard_count)
        THROW("Shard is should be smaller than shard count")
    _loader_module->set_output(_outputs[0]);
    // Set reader and decoder config accordingly for the CIFAR10LoaderSingleShardNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, "", std::map<std::string, std::string>(), shuffle, loop);
    reader_cfg.set_shard_count(shard_count);
    reader_cfg.set_shard_id(shard_id);
    reader_cfg.set_file_prefix(file_prefix);
    reader_cfg.set_sharding_info(sharding_info);
    reader_cfg.set_batch_count(load_batch_count);
    _loader_module->initialize(reader_cfg, DecoderConfig(DecoderType::SKIP_DECODE), mem_type, _batch_size);
    _loader_module->start_loading();

    std::array<std::string, INIT_ARGS_COUNT> arg_names = {
        "shard_id", "shard_count", "source_path", "storage_type",
        "shuffle", "loop", "load_batch_count", "mem_type", "file_prefix",
        "last_batch_policy", "pad_last_batch_repeated", "stick_to_shard", "shard_size"
    };
    // NOTE : Add the new arguments when modifying init function
    set_node_arguments(arg_names, std::make_index_sequence<arg_names.size()>{}, shard_id, shard_count, source_path,
                       storage_type, shuffle, loop, load_batch_count, mem_type, file_prefix, sharding_info.last_batch_policy,
                       sharding_info.pad_last_batch_repeated, sharding_info.stick_to_shard, sharding_info.shard_size);
}

std::shared_ptr<LoaderModule> CIFAR10LoaderSingleShardNode::get_loader_module() {
    if (!_loader_module)
        THROW("CIFAR10LoaderSingleShardNode's loader module is null, not initialized")
    return _loader_module;
}

CIFAR10LoaderSingleShardNode::~CIFAR10LoaderSingleShardNode() {
    _loader_module = nullptr;
}

void CIFAR10LoaderSingleShardNode::initialize_args(std::vector<Argument> &arguments, std::shared_ptr<MetaDataReader> meta_data_reader) {
    (void)meta_data_reader;
    if (arguments.size() != INIT_ARGS_COUNT)
        THROW("CIFAR10LoaderSingleShardNode expected " + std::to_string(INIT_ARGS_COUNT) + " arguments, received " + std::to_string(arguments.size()) +
              ". Ensure all arguments present in init are accounted for");

    ShardingInfo sharding_info(arguments[9].get<RocalBatchPolicy>(), arguments[10].get<bool>(), arguments[11].get<bool>(), arguments[12].get<int32_t>());
    this->init(arguments[0].get<unsigned>(), arguments[1].get<unsigned>(), arguments[2].get<std::string>(),
               arguments[3].get<StorageType>(), arguments[4].get<bool>(), arguments[5].get<bool>(),
               arguments[6].get<size_t>(), arguments[7].get<RocalMemType>(), arguments[8].get<std::string>(), sharding_info);
}
