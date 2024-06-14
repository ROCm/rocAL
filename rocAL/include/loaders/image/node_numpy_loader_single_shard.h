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
#include "numpy_loader_sharded.h"
#include "pipeline/graph.h"
#include "pipeline/node.h"

class NumpyLoaderSingleShardNode : public Node {
   public:
    NumpyLoaderSingleShardNode(Tensor *output, void *device_resources);
    ~NumpyLoaderSingleShardNode() override;

    /// \param shard_id shard id from user
    /// \param shard_count shard count from user
    /// \param source_path Defines the path that includes the image dataset
    /// \param storage_type Determines the storage type
    /// \param decoder_type Determines the decoder_type
    /// \param shuffle Determines if the user wants to shuffle the dataset or not.
    /// \param loop Determines if the user wants to indefinitely loops through audios or not.
    /// \param load_batch_count Defines the quantum count of the Audios to be loaded. It's usually equal to the user's batch size.
    /// \param mem_type Memory type, host or device
    /// \param last_batch_policy Determines the handling of the last batch when the shard size is not divisible by the batch size.
    void init(unsigned shard_id, unsigned shard_count, const std::string &source_path,
              StorageType storage_type, DecoderType decoder_type, bool shuffle, bool loop,
              size_t load_batch_count, RocalMemType mem_type, std::pair<RocalBatchPolicy, bool> last_batch_info = {RocalBatchPolicy::FILL, true});

    std::shared_ptr<LoaderModule> get_loader_module();

   protected:
    void create_node() override {};
    void update_node() override {};

   private:
    std::shared_ptr<NumpyLoader> _loader_module = nullptr;
};
