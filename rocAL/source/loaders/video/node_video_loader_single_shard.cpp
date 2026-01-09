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

#include "loaders/video/node_video_loader_single_shard.h"

#include <array>
#include "pipeline/exception.h"
#include "readers/video/video_properties.h"
#ifdef ROCAL_VIDEO

#define INIT_ARGS_COUNT 14  // Modify in accordance with number of args in init

REGISTER_LOADER_NODE(VideoLoaderSingleShardNode)

VideoLoaderSingleShardNode::VideoLoaderSingleShardNode(Tensor *output, void *device_resources) : Node({}, {output}) {
    _loader_module = std::make_shared<VideoLoader>(device_resources);
}

void VideoLoaderSingleShardNode::init(unsigned shard_id, unsigned shard_count, const std::string &source_path, StorageType storage_type, DecoderType decoder_type, DecodeMode decoder_mode,
                                      unsigned sequence_length, unsigned step, unsigned stride, VideoProperties &video_prop, bool shuffle, bool loop, size_t load_batch_count, RocalMemType mem_type) {
    _decode_mode = decoder_mode;  // for future use
    if (!_loader_module)
        THROW("ERROR: loader module is not set for VideoLoaderNode, cannot initialize")
    if (shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    if (shard_id >= shard_count)
        THROW("Shard is should be smaller than shard count")
    _loader_module->set_output(_outputs[0]);
    // Set reader and decoder config accordingly for the ImageLoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, "", std::map<std::string, std::string>(), shuffle, loop);
    reader_cfg.set_shard_count(shard_count);
    reader_cfg.set_shard_id(shard_id);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_sequence_length(sequence_length);
    reader_cfg.set_frame_step(step);
    reader_cfg.set_frame_stride(stride);
    reader_cfg.set_video_properties(video_prop);

    std::array<std::string, INIT_ARGS_COUNT> arg_names = {
        "shard_id", "shard_count", "source_path", "storage_type", "decoder_type",
        "decoder_mode", "sequence_length", "step", "stride", "file_list_frame_num",
        "shuffle", "loop", "load_batch_count", "mem_type"};

    // NOTE : Update arg_names when modifying init
    set_node_arguments(arg_names, std::make_index_sequence<arg_names.size()>{}, shard_id, shard_count, source_path, storage_type, decoder_type,
                       decoder_mode, sequence_length, step, stride, video_prop.file_list_frame_num, shuffle, loop, load_batch_count, mem_type);
    _loader_module->initialize(reader_cfg, DecoderConfig(decoder_type), mem_type, _batch_size);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> VideoLoaderSingleShardNode::get_loader_module() {
    if (!_loader_module)
        WRN("VideoLoaderSingleShardNode's loader module is null, not initialized")
    return _loader_module;
}

VideoLoaderSingleShardNode::~VideoLoaderSingleShardNode() {
    _loader_module = nullptr;
}

void VideoLoaderSingleShardNode::initialize_args(std::vector<Argument> &arguments, std::shared_ptr<MetaDataReader> meta_data_reader) {
    (void)meta_data_reader;
    if (arguments.size() != INIT_ARGS_COUNT)
        THROW("VideoLoaderSingleShardNode expected " + std::to_string(INIT_ARGS_COUNT) + " arguments, received " + std::to_string(arguments.size()) +
              "Ensure all arguments present in init are accounted for");

    auto source_path = arguments[2].get<std::string>();
    auto file_list_frame_num = arguments[9].get<bool>();

    VideoProperties video_prop;
    find_video_properties(video_prop, source_path.c_str(), file_list_frame_num);

    this->init(arguments[0].get<unsigned>(), arguments[1].get<unsigned>(), source_path, arguments[3].get<StorageType>(), arguments[4].get<DecoderType>(),
               arguments[5].get<DecodeMode>(), arguments[6].get<unsigned>(), arguments[7].get<unsigned>(), arguments[8].get<unsigned>(), video_prop,
               arguments[10].get<bool>(), arguments[11].get<bool>(), arguments[12].get<size_t>(), arguments[13].get<RocalMemType>());
}
#endif
