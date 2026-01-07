/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "loaders/image/node_numpy_loader_single_shard.h"

#include "pipeline/exception.h"

#define INIT_ARGS_COUNT 15  // Modify in accordance with number of args in init

REGISTER_LOADER_NODE(NumpyLoaderSingleShardNode)

NumpyLoaderSingleShardNode::NumpyLoaderSingleShardNode(Tensor *output, void *device_resources) : Node({}, {output}) {
    _loader_module = std::make_shared<NumpyLoader>(device_resources);
}

void NumpyLoaderSingleShardNode::init(unsigned shard_id, unsigned shard_count, const std::string &source_path, const std::vector<std::string> &files, StorageType storage_type, DecoderType decoder_type,
                                      bool shuffle, bool loop, size_t load_batch_count, RocalMemType mem_type, unsigned seed, const ShardingInfo& sharding_info) {
    if (!_loader_module)
        THROW("ERROR: loader module is not set for NumpyLoaderSingleShardNode, cannot initialize")
    if (shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    if (shard_id >= shard_count)
        THROW("Shard is should be smaller than shard count")
    _loader_module->set_output(_outputs[0]);
    // Set reader and decoder config accordingly for the NumpyLoaderSingleShardNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, "", std::map<std::string, std::string>(), shuffle, loop);
    reader_cfg.set_shard_count(shard_count);
    reader_cfg.set_shard_id(shard_id);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_files_list(files);
    reader_cfg.set_seed(seed);
    reader_cfg.set_sharding_info(sharding_info);
    _loader_module->initialize(reader_cfg, DecoderConfig(DecoderType::SKIP_DECODE), mem_type, _batch_size);
    _loader_module->start_loading();

    std::array<std::string, INIT_ARGS_COUNT> arg_names = {
        "shard_id", "shard_count", "source_path", "files", "storage_type", "decoder_type",
        "shuffle", "loop", "load_batch_count", "mem_type", "seed", "last_batch_policy",
        "pad_last_batch_repeated", "stick_to_shard", "shard_size"
    };
    // NOTE : Add the new arguments when modifying init function
    set_node_arguments(arg_names, std::make_index_sequence<arg_names.size()>{}, shard_id, shard_count, source_path, files,
                       storage_type, decoder_type, shuffle, loop, load_batch_count, mem_type, seed, sharding_info.last_batch_policy,
                       sharding_info.pad_last_batch_repeated, sharding_info.stick_to_shard, sharding_info.shard_size);
}

std::shared_ptr<LoaderModule> NumpyLoaderSingleShardNode::get_loader_module() {
    if (!_loader_module)
        WRN("NumpyLoaderSingleShardNode's loader module is null, not initialized")
    return _loader_module;
}

NumpyLoaderSingleShardNode::~NumpyLoaderSingleShardNode() {
    _loader_module = nullptr;
}

void NumpyLoaderSingleShardNode::initialize_args(std::vector<Argument> &arguments, std::shared_ptr<MetaDataReader> meta_data_reader) {
    (void)meta_data_reader;
    if (arguments.size() != INIT_ARGS_COUNT)
        THROW("NumpyLoaderSingleShardNode expected " + std::to_string(INIT_ARGS_COUNT) + " arguments, received " + std::to_string(arguments.size()) +
              ". Ensure all arguments present in init are accounted for");

    ShardingInfo sharding_info(arguments[11].get<RocalBatchPolicy>(), arguments[12].get<bool>(), arguments[13].get<bool>(), arguments[14].get<int32_t>());
    this->init(arguments[0].get<unsigned>(), arguments[1].get<unsigned>(), arguments[2].get<std::string>(),
               arguments[3].get<std::vector<std::string>>(), arguments[4].get<StorageType>(), arguments[5].get<DecoderType>(),
               arguments[6].get<bool>(), arguments[7].get<bool>(), arguments[8].get<size_t>(), arguments[9].get<RocalMemType>(),
               arguments[10].get<unsigned>(), sharding_info);
}
