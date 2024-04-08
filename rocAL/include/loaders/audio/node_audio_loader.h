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
#include "audio_loader_sharded.h"
#include "graph.h"
#include "node.h"

#ifdef ROCAL_AUDIO

class AudioLoaderNode : public Node {
   public:
    /// \param device_resources shard count from user
    /// internal_shard_count number of loader/decoders are created and each shard is loaded and decoded using separate and independent resources increasing the parallelism and performance.
    AudioLoaderNode(Tensor *output, void *device_resources);
    ~AudioLoaderNode() override;
    AudioLoaderNode() = delete;
    /// \param internal_shard_count Defines the amount of parallelism user wants for the load and decode process to be handled internally.
    /// \param source_path Defines the path that includes the Audio dataset
    /// \param file_list_path Defines the path that contains the file list
    /// \param storage_type Determines the storage type
    /// \param decoder_type Determines the decoder_type
    /// \param shuffle Determines if the user wants to shuffle the dataset or not.
    /// \param loop Determines if the user wants to indefinitely loops through audio or not.
    /// \param load_batch_count Defines the quantum count of the Audios to be loaded. It's usually equal to the user's batch size.
    /// \param mem_type Memory type, host or device
    /// \param meta_data_reader Determines the meta-data information
    /// \param stick_to_shard Determines after each epoch if the pipeline advances to the next shard to increase the entropy of the data that is seen by this pipeline or not.
    /// \param shard_size Number of samples in the shard for the wrapped pipeline. Providing -1 means that the iterator will work until StopIteration is raised from the inside of iterator.
    /// The loader will repeat Audios if necessary to be able to have Audios in multiples of the load_batch_count,
    /// for example if there are 10 Audios in the dataset and load_batch_count is 3, the loader repeats 2 Audios as if there are 12 Audios available.
    void Init(unsigned internal_shard_count, unsigned cpu_num_threads, const std::string &source_path, const std::string &file_list_path, 
              StorageType storage_type, DecoderType decoder_type, bool shuffle, bool loop, size_t load_batch_count, 
              RocalMemType mem_type, std::shared_ptr<MetaDataReader> meta_data_reader,
              RocalBatchPolicy last_batch_policy = RocalBatchPolicy::FILL, bool last_batch_padded = false, bool stick_to_shard = false, signed shard_size = -1);
    std::shared_ptr<LoaderModule> GetLoaderModule();

   protected:
    void create_node() override{};
    void update_node() override{};

   private:
    std::shared_ptr<AudioLoaderSharded> _loader_module = nullptr;
};
#endif
