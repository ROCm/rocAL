/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#include <dirent.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "meta_data/meta_data_graph.h"
#include "meta_data/meta_data_reader.h"
#include "readers/image/image_reader.h"
#include "pipeline/timing_debug.h"

class COCOFileSourceReader : public Reader {
   public:
    //! Looks up the folder which contains the files, amd loads the image names
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig desc) override;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded images
     \return Size of the loaded resource
    */
    size_t read_data(unsigned char *buf, size_t max_size) override;
    //! Opens the next file in the folder
    /*!
     \return The size of the next file, 0 if couldn't access it
    */
    size_t open() override;

    //! Resets the object's state to read from the first file in the folder
    void reset() override;

    //! Returns the name of the latest file opened
    std::string id() override { return _last_id; };

    unsigned count_items() override;

    ~COCOFileSourceReader() override;

    int close() override;

    COCOFileSourceReader();

    size_t last_batch_padded_size() override; // The size of the number of samples padded in the last batch

   private:
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    //! opens the folder containnig the images
    Reader::Status open_folder();
    Reader::Status subfolder_reading();
    std::string _folder_path;
    std::string _json_path;
    DIR *_src_dir;
    DIR *_sub_dir;
    struct dirent *_entity;
    std::vector<std::string> _file_names, _sorted_file_names;
    std::vector<float> _aspect_ratios;
    unsigned _curr_file_idx;
    FILE *_current_fPtr;
    std::ifstream _current_ifs;
    unsigned _current_file_size;
    std::string _last_id;
    std::string _last_file_name;
    size_t _shard_id = 0;
    size_t _shard_count = 1;  // equivalent of batch size
    size_t _batch_size = 1;
    bool _loop;
    bool _shuffle;
    int _read_counter = 0;
    //!< _file_count_all_shards total_number of files in to figure out the max_batch_size (usually needed for distributed training).
    size_t _file_count_all_shards;
    void incremenet_read_ptr();
    int release();
    void shuffle_with_aspect_ratios();
    void increment_curr_file_idx();
    ShardingInfo _sharding_info = ShardingInfo();  // The members of ShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
    size_t _last_batch_padded_size = 0;
    bool _stick_to_shard = false;
    bool _pad_last_batch_repeated = false;
    int32_t _shard_size = -1;
    std::vector<unsigned> _shard_start_idx_vector, _shard_end_idx_vector;
    std::vector<std::string> _all_shard_file_names_padded;
    Reader::Status generate_file_names(); // Function that would generate _file_names containing all the samples in the dataset
    size_t get_dataset_size(); // DataSet Size
    size_t actual_shard_size_without_padding(); // Number of files belonging to a shard (without padding)
    size_t largest_shard_size_without_padding(); // Number of files belonging to a shard (with padding)
    //!< Used to advance to the next shard's data to increase the entropy of the data seen by the pipeline>
    void increment_shard_id();
    void compute_start_and_end_idx_of_all_shards();     // Start Idx of all the Shards
};
