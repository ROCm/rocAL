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

#include <memory>
#include <string>
#include <vector>

#include "pipeline/commons.h"
#include "pipeline/timing_debug.h"
#include "readers/image/image_reader.h"

class FileSourceReader : public Reader {
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

    //! Returns the name of the latest file_path opened
    const std::string file_path() override { return _last_file_path; }

    unsigned count_items() override;

    ~FileSourceReader() override;

    int close() override;

    FileSourceReader();

    size_t last_batch_padded_size() override;  // The size of the number of samples padded in the last batch

    std::string get_root_folder_path() override;  // Returns the root folder path

    std::vector<std::string> get_file_paths_from_meta_data_reader() override;  // Returns the relative file path from the meta-data reader
   private:
    //! opens the folder containnig the images
    Reader::Status open_folder();
    Reader::Status subfolder_reading();
    std::string _folder_path;
    std::string _file_list_path;
    DIR *_src_dir;
    DIR *_sub_dir;
    struct dirent *_entity;
    std::vector<std::string> _file_names;
    unsigned _curr_file_idx;
    FILE *_current_fPtr;
    unsigned _current_file_size;
    unsigned _shard_start_idx;
    std::vector<unsigned> _shard_start_idx_vector, _shard_end_idx_vector;
    std::string _last_id;
    std::string _last_file_name, _last_file_path, _absolute_file_path;
    size_t _shard_id = 0;
    size_t _shard_count = 1;  // equivalent of batch size
    int32_t _shard_size = -1;
    size_t _batch_size = 1;
    size_t _file_id = 0;
    size_t _padded_samples = 0;
    bool _loop;
    bool _shuffle;
    int _read_counter = 0;
    //!< _file_count_all_shards total_number of files in to figure out the max_batch_size (usually needed for distributed training).
    size_t _file_count_all_shards;
    void incremenet_read_ptr();
    void increment_curr_file_idx();
    int release();
    size_t get_file_shard_id();
    void incremenet_file_id() { _file_id++; }
    void fill_last_batch();
    void replicate_last_batch_to_pad_partial_shard();
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    //! Pair containing the last batch policy and pad_last_batch_repeated values for deciding what to do with last batch
    ShardingInfo _last_batch_info = ShardingInfo();  //  A std::pair object representing the Last Batch Policies in rocAL and the padding of the samples.
                                                         //  first: Determines the handling of the last batch when the shard size is not divisible by the batch size. Check RocalLastBatchPolicy() enum for possible values.
                                                         //  second: If set to True, pads the shards last batch by repeating the last sample's data (dummy data).
    size_t _last_batch_padded_size = 0;                  // The size of number of padded samples in the last batch
    size_t _num_padded_samples = 0;                      //! Number of samples that are padded in the last batch which would differ for each shard.
    bool _stick_to_shard = false;
    bool _pad_last_batch_repeated = false;
    Reader::Status generate_file_names();         // Function that would generate _file_names containing all the samples in the dataset
    void compute_start_and_end_idx_of_all_shards();     // Start Idx of all the Shards
    size_t get_dataset_size();                    // DataSet Size
    size_t actual_shard_size_without_padding();   // Actual Number of Files present in the shard (without padding)
    size_t largest_shard_size_without_padding();  // The size of the shard having largest files (without padding)
    //!< Used to advance to the next shard's data to increase the entropy of the data seen by the pipeline>
    void increment_shard_id();
};
