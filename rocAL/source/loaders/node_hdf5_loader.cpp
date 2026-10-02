/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "loaders/node_hdf5_loader.h"

Hdf5LoaderNode::Hdf5LoaderNode(const std::vector<Tensor*>& outputs, void* device_resources)
    : Node({}, outputs), _loader_module(std::make_shared<Hdf5Loader>(device_resources)) {}

Hdf5LoaderNode::~Hdf5LoaderNode() {
    _loader_module = nullptr;
}

void Hdf5LoaderNode::init(unsigned shard_id, unsigned shard_count,
                          const std::string& source_path,
                          const std::vector<std::string>& files,
                          const std::vector<std::string>& dataset_keys,
                          bool shuffle, bool loop, size_t batch_size,
                          RocalMemType mem_type, unsigned seed,
                          const ShardingInfo& sharding_info) {
    _loader_module->set_outputs(_outputs);
    ReaderConfig reader_config(StorageType::HDF5_DATA, source_path, "", {}, shuffle, loop);
    reader_config.set_shard_id(shard_id);
    reader_config.set_shard_count(shard_count);
    reader_config.set_batch_count(batch_size);
    reader_config.set_files_list(files);
    reader_config.set_dataset_keys(dataset_keys);
    reader_config.set_seed(seed);
    reader_config.set_sharding_info(sharding_info);
    _loader_module->initialize(reader_config, DecoderConfig(DecoderType::SKIP_DECODE),
                               mem_type, batch_size);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> Hdf5LoaderNode::get_loader_module() {
    return _loader_module;
}
