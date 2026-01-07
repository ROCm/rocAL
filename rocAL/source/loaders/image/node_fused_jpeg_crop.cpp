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

#include "loaders/image/node_fused_jpeg_crop.h"

#include "pipeline/exception.h"

#define INIT_ARGS_COUNT 18  // Modify in accordance with number of args in init

REGISTER_LOADER_NODE(FusedJpegCropNode)

FusedJpegCropNode::FusedJpegCropNode(Tensor *output, void *device_resources) : Node({}, {output}) {
    _loader_module = std::make_shared<ImageLoaderSharded>(device_resources);
}

void FusedJpegCropNode::init(unsigned internal_shard_count, unsigned cpu_num_threads, const std::string &source_path, const std::string &json_path, StorageType storage_type,
                             DecoderType decoder_type, bool shuffle, bool loop, size_t load_batch_count, RocalMemType mem_type, std::shared_ptr<MetaDataReader> meta_data_reader,
                             unsigned num_attempts, std::vector<float> &random_area, std::vector<float> &random_aspect_ratio, const ShardingInfo& sharding_info) {
    if (!_loader_module)
        THROW("ERROR: loader module is not set for FusedJpegCropNode, cannot initialize")
    if (internal_shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    _loader_module->set_output(_outputs[0]);
    // Set reader and decoder config accordingly for the FusedJpegCropNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, json_path, std::map<std::string, std::string>(), shuffle, loop);
    reader_cfg.set_shard_count(internal_shard_count);
    reader_cfg.set_cpu_num_threads(cpu_num_threads);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_meta_data_reader(meta_data_reader);
    reader_cfg.set_sharding_info(sharding_info);
    auto decoder_cfg = DecoderConfig(decoder_type);

    decoder_cfg.set_random_area(random_area);
    decoder_cfg.set_random_aspect_ratio(random_aspect_ratio);
    decoder_cfg.set_num_attempts(num_attempts);
    decoder_cfg.set_seed(ParameterFactory::instance()->get_seed());
    _loader_module->initialize(reader_cfg, decoder_cfg,
                               mem_type,
                               _batch_size);
    _loader_module->start_loading();

    std::array<std::string, INIT_ARGS_COUNT> arg_names = {
        "internal_shard_count", "cpu_num_threads", "source_path", "json_path", "storage_type",
        "decoder_type", "shuffle", "loop", "load_batch_count", "mem_type", "meta_data_reader",
        "num_attempts", "random_area", "random_aspect_ratio", "last_batch_policy",
        "pad_last_batch_repeated", "stick_to_shard", "shard_size"
    };
    // NOTE: Update arg_names when modifying init
    set_node_arguments(arg_names, std::make_index_sequence<arg_names.size()>{}, internal_shard_count, cpu_num_threads,
                       source_path, json_path, storage_type, decoder_type, shuffle, loop, load_batch_count, mem_type,
                       meta_data_reader, num_attempts, random_area, random_aspect_ratio, sharding_info.last_batch_policy,
                       sharding_info.pad_last_batch_repeated, sharding_info.stick_to_shard, sharding_info.shard_size);
}

std::shared_ptr<LoaderModule> FusedJpegCropNode::get_loader_module() {
    if (!_loader_module)
        WRN("FusedJpegCropNode's loader module is null, not initialized")
    return _loader_module;
}

FusedJpegCropNode::~FusedJpegCropNode() {
    _loader_module = nullptr;
}

void FusedJpegCropNode::initialize_args(std::vector<Argument> &arguments, std::shared_ptr<MetaDataReader> meta_data_reader) {
    if (arguments.size() != INIT_ARGS_COUNT)
        THROW("FusedJpegCropNode expected " + std::to_string(INIT_ARGS_COUNT) + " arguments, received " + std::to_string(arguments.size()) +
              ". Ensure all arguments present in init are accounted for");

    auto random_area = arguments[12].get<std::vector<float>>();
    auto random_aspect_ratio = arguments[13].get<std::vector<float>>();
    ShardingInfo sharding_info(arguments[14].get<RocalBatchPolicy>(), arguments[15].get<bool>(), arguments[16].get<bool>(), arguments[17].get<int32_t>());

    this->init(arguments[0].get<unsigned>(), arguments[1].get<unsigned>(), arguments[2].get<std::string>(),
               arguments[3].get<std::string>(), arguments[4].get<StorageType>(), arguments[5].get<DecoderType>(),
               arguments[6].get<bool>(), arguments[7].get<bool>(), arguments[8].get<size_t>(), arguments[9].get<RocalMemType>(),
               meta_data_reader, arguments[11].get<unsigned>(), random_area, random_aspect_ratio, sharding_info);
}
