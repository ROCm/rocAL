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
#include <dirent.h>
#include <fcntl.h>
#include <libtar.h>
#include <unistd.h>

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "meta_data/webdataset_meta_data_reader.h"
#include "pipeline/timing_debug.h"
#include "readers/image/image_reader.h"

class WebDatasetSourceReader : public Reader {
  public:
    //! Reads the TFRecord File, and loads the image ids and other necessary
    //! info
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

    //! Returns the id of the latest file opened
    std::string id() override { return _last_id; };

    unsigned count_items() override;

    ~WebDatasetSourceReader() override;

    int close() override;

    WebDatasetSourceReader();

  private:
    //! opens the folder containnig the images
    Reader::Status webdataset_record_reader();
    Reader::Status folder_reading();
    std::string _folder_path;
    std::string _path;
    std::string _paths, _index_paths;
    std::map<std::string, std::string> _feature_key_map;
    std::string _encoded_key;
    std::string _filename_key;
    DIR *_src_dir;
    DIR *_sub_dir;
    struct dirent *_entity;
    std::vector<std::string> _file_names;
    std::map<std::string, unsigned int> _file_wds_shard_idx_mapping;
    std::map<std::string, unsigned int> _file_size, _file_offset;
    unsigned _curr_file_idx;
    unsigned _current_file_size;
    std::string _last_id;
    std::string _last_file_name;
    unsigned int _last_file_size;
    std::vector<std::string> _index_name_list;
    size_t _shard_id = 0;
    size_t _shard_count = 1; // equivalent of batch size
    int _file_name_count = 0;
    bool _last_rec;
    //!< _batch_count Defines the quantum count of the images to be read. It's
    //!< usually equal to the user's batch size.
    /// The loader will repeat images if necessary to be able to have images
    /// available in multiples of the load_batch_count, for instance if there
    /// are 10 images in the dataset and _batch_count is 3, the loader repeats 2
    /// images as if there are 12 images available.
    size_t _batch_count = 1;
    size_t _file_id = 0;
    size_t _in_batch_read_count = 0;
    bool _loop;
    bool _shuffle;
    int _read_counter = 0;
    size_t _file_count_all_shards;
    void incremenet_read_ptr();
    int release();
    void incremenet_file_id() { _file_id++; }
    void parse_tar_files(std::vector<SampleDescription> &samples_container, std::vector<ComponentDescription> &components_container, std::unique_ptr<FileIOStream> &tar_file);
    void parse_index_files(std::vector<SampleDescription> &samples_container,
                      std::vector<ComponentDescription> &components_container,
                      const std::string &index_path);
    void parse_sample_description(
        std::vector<SampleDescription> &samples_container,
        std::vector<ComponentDescription> &components_container,
        std::ifstream &index_file, const std::string &index_path, int64_t line,
        int index_version);
    Reader::Status webdataset_record_reader_from_components(ComponentDescription component, unsigned wds_shard_index);
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    std::vector<std::unique_ptr<FileIOStream>> _wds_shards;
    Reader::Status read_web_dataset_at_offset(unsigned char *buff,
                                              std::string file_name,
                                              uint file_size, uint offset,
                                              uint wds_shard_index);
};
