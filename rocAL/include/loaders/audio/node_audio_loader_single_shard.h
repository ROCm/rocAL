/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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
#include "loaders/audio/audio_loader_sharded.h"
#include "pipeline/graph.h"
#include "pipeline/node.h"

#ifdef ROCAL_AUDIO

class AudioLoaderSingleShardNode : public Node {
   public:
    AudioLoaderSingleShardNode(Tensor *output, void *device_resources);
    ~AudioLoaderSingleShardNode() override;
    /// \param user_shard_count shard count from user
    /// \param  user_shard_id shard id from user
    /// \param source_path Defines the path that includes the Audio dataset
    /// \param file_list_path Defines the path that contains the file list
    /// \param storage_type Determines the storage type
    /// \param decoder_type Determines the decoder_type
    /// \param shuffle Determines if the user wants to shuffle the dataset or not.
    /// \param loop Determines if the user wants to indefinitely loops through audios or not.
    /// \param load_batch_count Defines the quantum count of the Audios to be loaded. It's usually equal to the user's batch size.
    /// \param mem_type Memory type, host or device
    /// \param meta_data_reader Determines the meta-data information
    /// \param last_batch_policy, pad_last_batch_repeated A std::pair object representing the Last Batch Policies in rocAL and the padding of the samples.
    ///            first: Determines the handling of the last batch when the shard size is not divisible by the batch size. Check RocalLastBatchPolicy() enum for possible values.
    ///            second: If set to True, pads the shards last batch by repeating the last sample's data (dummy data).
    /// \param stick_to_shard Determines whether reader should stick to a single shards dataset or it to be used in a round robin fashion.
    /// \param shard_size Provides the size of the shard for an epoch.
    /// The loader will repeat Audios if necessary to be able to have Audios in multiples of the load_batch_count,
    /// for example if there are 10 Audios in the dataset and load_batch_count is 3, the loader repeats 2 Audios as if there are 12 Audios available.
    void Init(unsigned shard_id, unsigned shard_count, unsigned cpu_num_threads, const std::string &source_path,
              const std::string &file_list_path, StorageType storage_type, DecoderType decoder_type, bool shuffle,
              bool loop, size_t load_batch_count, RocalMemType mem_type, std::shared_ptr<MetaDataReader> meta_data_reader,
              std::pair<RocalBatchPolicy, bool> last_batch_info = {RocalBatchPolicy::FILL, false}, bool stick_to_shard = false, signed shard_size = -1);
    std::shared_ptr<LoaderModule> GetLoaderModule();

   protected:
    void create_node() override{};
    void update_node() override{};

   private:
    std::shared_ptr<AudioLoader> _loader_module = nullptr;
};
#endif
