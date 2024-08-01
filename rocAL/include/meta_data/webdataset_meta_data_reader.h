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

#include <map>

#include "meta_data/meta_data.h"
#include "meta_data/meta_data_reader.h"
#include "pipeline/commons.h"
#include "pipeline/filesystem.h"
#include "readers/image/image_reader.h"
#include "tar_helper_functions.h"
#include <set>
#include <unordered_set>
class WebDataSetMetaDataReader : public MetaDataReader {
  public:
    void init(const MetaDataConfig &cfg,
              pMetaDataBatch meta_data_batch) override;
    void lookup(const std::vector<std::string> &image_names) override;
    void read_all(const std::string &path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    bool set_timestamp_mode() override { return false; }
    const std::map<std::string, std::shared_ptr<MetaData>> &
    get_map_content() override {
        return _map_content;
    }
    WebDataSetMetaDataReader();

  private:
    bool exists(const std::string &image_name) override;
    void add(std::string image_name, int label);
    std::map<std::string, std::shared_ptr<MetaData>> _map_content;
    std::map<std::string, std::shared_ptr<MetaData>>::iterator _itr;
    std::string _path;
    std::string _paths, _index_paths;
    std::vector<std::string> _index_name_list;
    MissingComponentsBehaviour _missing_component_behaviour;
    pMetaDataBatch _output;
    DIR *_src_dir, *_sub_dir;
    struct dirent *_entity;
    int _file_name_count = 0;
    std::vector<std::string> _file_names;
    std::vector<std::set<std::string>> _exts, _meta_data_exts;
    std::unordered_map<std::string, uint> _ext_map;
    std::vector<std::string> _subfolder_file_names;
    void parse_tar_files(std::vector<SampleDescription> &samples_container,
                    std::vector<ComponentDescription> &components_container,
                    std::unique_ptr<FileIOStream> &tar_file);
    void parse_index_files(std::vector<SampleDescription> &samples_container,
                      std::vector<ComponentDescription> &components_container,
                      const std::string &paths_to_index_files);
    void parse_sample_description(
        std::vector<SampleDescription> &samples_container,
        std::vector<ComponentDescription> &components_container,
        std::ifstream &index_file, const std::string &index_path, int64_t line,
        int index_version);
    void read_sample_and_add_to_map(
        ComponentDescription component,
        std::unique_ptr<FileIOStream> &current_tar_file_stream,
        AsciiValues ascii_values);
    void add(std::string image_name, AsciiValues ascii_value);
    std::vector<std::unique_ptr<FileIOStream>> _wds_shards;
};
