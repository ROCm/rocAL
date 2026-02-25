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

#include "loaders/image/node_image_loader.h"

#include "pipeline/exception.h"
#include "rocal.pb.h"

REGISTER_LOADER_NODE(ImageLoaderNode)

ImageLoaderNode::ImageLoaderNode(Tensor *output, void *device_resources) : Node({}, {output}) {
    _loader_module = std::make_shared<ImageLoaderSharded>(device_resources);
}

void ImageLoaderNode::init(unsigned internal_shard_count, unsigned cpu_num_threads, const std::string &source_path, const std::string &json_path, const std::map<std::string, std::string> feature_key_map, StorageType storage_type, DecoderType decoder_type,
                           bool shuffle, bool loop, size_t load_batch_count, RocalMemType mem_type, std::shared_ptr<MetaDataReader> meta_data_reader, bool decoder_keep_orig, const ShardingInfo& sharding_info, bool enable_checkpointing, unsigned seed, const char *file_prefix,
                           unsigned sequence_length, unsigned step, unsigned stride, ExternalSourceFileMode external_file_mode, const std::string &index_path) {
    if (!_loader_module)
        THROW("ERROR: loader module is not set for ImageLoaderNode, cannot initialize")
    if (internal_shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    _loader_module->set_output(_outputs[0]);
    // Set reader and decoder config accordingly for the ImageLoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, json_path, feature_key_map, shuffle, loop);
    reader_cfg.set_shard_count(internal_shard_count);
    reader_cfg.set_cpu_num_threads(cpu_num_threads);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_file_prefix(file_prefix);
    reader_cfg.set_meta_data_reader(meta_data_reader);
    //  sequence_length, step and stride parameters used only for SequenceReader
    reader_cfg.set_sequence_length(sequence_length);
    reader_cfg.set_frame_step(step);
    reader_cfg.set_frame_stride(stride);
    reader_cfg.set_external_filemode(external_file_mode);
    reader_cfg.set_index_path(index_path);
    reader_cfg.set_sharding_info(sharding_info);
    reader_cfg.enable_checkpointing(enable_checkpointing);
    reader_cfg.set_seed(seed);

    // Add arguments to ArgumentSet one by one
    _args.add_new_argument("internal_shard_count", internal_shard_count);
    _args.add_new_argument("cpu_num_threads", cpu_num_threads);
    _args.add_new_argument("source_path", source_path);
    _args.add_new_argument("json_path", json_path);
    _args.add_new_argument("feature_key_map", feature_key_map);
    _args.add_new_argument("storage_type", storage_type);
    _args.add_new_argument("decoder_type", decoder_type);
    _args.add_new_argument("shuffle", shuffle);
    _args.add_new_argument("loop", loop);
    _args.add_new_argument("load_batch_count", load_batch_count);
    _args.add_new_argument("mem_type", mem_type);
    _args.add_new_argument("meta_data_reader", meta_data_reader);
    _args.add_new_argument("decoder_keep_orig", decoder_keep_orig);
    _args.add_new_argument("last_batch_policy", sharding_info.last_batch_policy);
    _args.add_new_argument("pad_last_batch_repeated", sharding_info.pad_last_batch_repeated);
    _args.add_new_argument("stick_to_shard", sharding_info.stick_to_shard);
    _args.add_new_argument("shard_size", sharding_info.shard_size);
    _args.add_new_argument("enable_checkpointing", enable_checkpointing);
    _args.add_new_argument("seed", seed);
    _args.add_new_argument("file_prefix", file_prefix);
    _args.add_new_argument("sequence_length", sequence_length);
    _args.add_new_argument("step", step);
    _args.add_new_argument("stride", stride);
    _args.add_new_argument("external_file_mode", external_file_mode);
    _args.add_new_argument("index_path", index_path);

    _loader_module->initialize(reader_cfg, DecoderConfig(decoder_type),
                               mem_type,
                               _batch_size, decoder_keep_orig);
    _loader_module->start_loading();
}

void ImageLoaderNode::initialize_args(const ArgumentSet &arguments, std::shared_ptr<MetaDataReader> meta_data_reader) {
    ShardingInfo sharding_info(arguments.get<RocalBatchPolicy>("last_batch_policy"), arguments.get<bool>("stick_to_shard"), arguments.get<bool>("pad_last_batch_repeated"), arguments.get<int32_t>("shard_size"));
    std::string file_prefix = arguments.get<std::string>("file_prefix");

    // NOTE: Add respective arguments to init function in the same order as defined in init function
    this->init(arguments.get<unsigned>("internal_shard_count"), 
               arguments.get<unsigned>("cpu_num_threads"), 
               arguments.get<std::string>("source_path"),
               arguments.get<std::string>("json_path"), 
               arguments.get<std::map<std::string, std::string>>("feature_key_map"), 
               arguments.get<StorageType>("storage_type"),
               arguments.get<DecoderType>("decoder_type"), 
               arguments.get<bool>("shuffle"), 
               arguments.get<bool>("loop"), 
               arguments.get<size_t>("load_batch_count"), 
               arguments.get<RocalMemType>("mem_type"),
               meta_data_reader, arguments.get<bool>("decoder_keep_orig"), 
               sharding_info, arguments.get<bool>("enable_checkpointing"),
               arguments.get<unsigned>("seed"), file_prefix.c_str(),
               arguments.get<unsigned>("sequence_length"), 
               arguments.get<unsigned>("step"), 
               arguments.get<unsigned>("stride"), 
               arguments.get<ExternalSourceFileMode>("external_file_mode"),
               arguments.get<std::string>("index_path"));
}

std::shared_ptr<LoaderModule> ImageLoaderNode::get_loader_module() {
    if (!_loader_module)
        WRN("ImageLoaderNode's loader module is null, not initialized")
    return _loader_module;
}

ImageLoaderNode::~ImageLoaderNode() {
    _loader_module = nullptr;
}

// Capture the loader's current state into the operator checkpoint.
void ImageLoaderNode::save_state(std::shared_ptr<OperatorCheckpoint>& op_ckpt) {
    op_ckpt->GetMutableCheckpointState() = _loader_module->get_loader_state();
}

// Serialize loader state into a protobuf payload for checkpointing.
std::string ImageLoaderNode::serialize_state(const std::shared_ptr<OperatorCheckpoint>& op_ckpt) {
    auto loader_state = op_ckpt->GetOperatorCheckpointState<LoaderState>();
    rocal_proto::LoaderState proto_state;
    proto_state.set_current_epoch(static_cast<int32_t>(loader_state.epoch_number));
    proto_state.set_iteration_number(static_cast<int64_t>(loader_state.iteration_number));
    proto_state.set_rng(SerializeRNGToString(loader_state.rng));
    proto_state.set_curr_file_idx(static_cast<uint32_t>(loader_state.curr_file_idx));
    return proto_state.SerializeAsString();
}

// Restore loader state from a serialized checkpoint payload.
void ImageLoaderNode::restore_state(const std::string &operator_state_bytes) {
    rocal_proto::LoaderState proto_state;  // Protobuf loader state payload.
    if (!proto_state.ParseFromString(operator_state_bytes)) {
        WRN("Failed to parse LoaderState from checkpoint. Skipping restore for ImageLoaderNode.");
        return;
    }
    LoaderState st{};  // Reconstructed loader state from checkpoint data.
    st.epoch_number = proto_state.has_current_epoch() ? proto_state.current_epoch() : 0;
    st.iteration_number = proto_state.has_iteration_number() ? proto_state.iteration_number() : 0;
    if (proto_state.has_rng()) {
        DeserializeRNGFromString(proto_state.rng(), st.rng);
    }
    // For sharded loaders, curr_file_idx is relative to each shard's file list.
    st.curr_file_idx = proto_state.has_curr_file_idx() ? proto_state.curr_file_idx() : 0;
    if (_loader_module) {
        _loader_module->restore_from_state(st);
    }
}
