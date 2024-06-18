
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

#include "readers/webdataset_source_reader.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define BLOCKSIZE 10240

WebDatasetSourceReader::WebDatasetSourceReader() {
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _loop = false;
    _shuffle = false;
    _file_id = 0;
    _last_rec = false;
    _file_count_all_shards = 0;
}

unsigned WebDatasetSourceReader::count_items() {
    if (_loop)
        return _file_names.size();

    int ret = ((int)_file_names.size() - _read_counter);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status WebDatasetSourceReader::initialize(ReaderConfig desc) {
    std::cerr << "\n WebDatasetSourceReader::initialize";
    auto ret = Reader::Status::OK;
    _file_id = 0;
    _folder_path = desc.path();
    _path = desc.path(); // TODO: need to add support fo the index file yet
    _index_paths = ""; // TODO: 
    _wds_shards.reserve(_path.size());
    _feature_key_map = desc.feature_key_map();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _loop = desc.loop();
    _meta_data_reader = desc.meta_data_reader();
    _shuffle = desc.shuffle();
    std::cerr << "\n Folder Reading";
    ret = folder_reading(); // TODO: Add support for reading from the index file if provided
    std::cerr << "\n file_names.size():: " << _file_names.size();
    std::cerr << "\n Folder Reading Done";
    // shuffle dataset if set
    if (ret == Reader::Status::OK && _shuffle)
        std::random_shuffle(_file_names.begin(), _file_names.end());
    return ret;
}

void WebDatasetSourceReader::incremenet_read_ptr() {
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
}
size_t WebDatasetSourceReader::open() {
    auto file_path = _file_names[_curr_file_idx];  // Get next file name
    _last_id = file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        _last_id.erase(0, last_slash_idx + 1);
    }
    _current_file_size = _file_size[_file_names[_curr_file_idx]];
    return _current_file_size;
}

size_t WebDatasetSourceReader::read_data(unsigned char *buf, size_t read_size) {
    auto ret = read_web_dataset_at_offset(buf, _file_names[_curr_file_idx], _file_size[_file_names[_curr_file_idx]], _file_offset[_file_names[_curr_file_idx]], _file_wds_shard_idx_mapping[_file_names[_curr_file_idx]]);
    if (ret != Reader::Status::OK)
        THROW("WebDatasetSourceReader: Error in reading tar records of the web  dataset reader");
    incremenet_read_ptr();
    return read_size;
}

int WebDatasetSourceReader::close() {
    return release();
}

WebDatasetSourceReader::~WebDatasetSourceReader() {
    release();
}

int WebDatasetSourceReader::release() {
    return 0;
}

void WebDatasetSourceReader::reset() {
    if (_shuffle)
        std::random_shuffle(_file_names.begin(), _file_names.end());
    _read_counter = 0;
    _curr_file_idx = 0;
}

std::tuple<std::string, std::string> WebDatasetSourceReader::split_name(const std::string& file_path) {
  size_t dot_pos = file_path.find('.', file_path.rfind('/') + 1);
  return {file_path.substr(0, dot_pos), file_path.substr(dot_pos + 1)};
}

void WebDatasetSourceReader::parse_tar_files(std::vector<SampleDescription>& samples_container,
                                              std::vector<ComponentDescription>& components_container,
                                              std::unique_ptr<StdFileStream>& tar_file) {
    int64_t initial_file_pos = tar_file->TellRead();
    std::cerr << "\n initial_file_pos" << initial_file_pos;
    TarArchive tar_archive(std::move(tar_file));

    std::string last_filename;
    for (; !tar_archive.EndOfArchive(); tar_archive.NextFile()) {
        if (tar_archive.GetFileType() == TarArchive::ENTRY_FILE) {
        std::tie(last_filename, std::ignore) = split_name(tar_archive.GetFileName());
        break;
        }
    }
    size_t last_components_size = components_container.size();
    for (; !tar_archive.EndOfArchive(); tar_archive.NextFile()) {
        if (tar_archive.GetFileType() != TarArchive::ENTRY_FILE) {
        continue;
        }

    std::string basename, ext;
    std::cerr << "\n tar_archive.GetFileName(): " << tar_archive.GetFileName();
    std::tie(basename, ext) = split_name(tar_archive.GetFileName());
    std::cerr << "\n basename: " << basename;
    std::cerr << "\n ext: " <<ext;
    if (basename.empty()) {
      continue;
    }

    if (basename != last_filename) {
      samples_container.emplace_back();
      samples_container.back().components = VectorView<ComponentDescription>(components_container, last_components_size, components_container.size() - last_components_size);
      last_filename = basename;
      last_components_size = components_container.size();
    }

    components_container.emplace_back();
    components_container.back().size = tar_archive.GetFileSize();
    components_container.back().offset = tar_archive.TellArchive() + tar_archive.HeaderSize();
    components_container.back().ext = std::move(ext);
    components_container.back().filename = basename;

  }
    samples_container.emplace_back();
    samples_container.back().components = VectorView<ComponentDescription>(components_container, last_components_size, components_container.size() - last_components_size);

    tar_file = tar_archive.Release();

}

Reader::Status WebDatasetSourceReader::folder_reading() {
    std::cerr << "\n WebDatasetSourceReader::folder_reading ";
    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;
    auto ret = Reader::Status::OK;

    while ((_entity = readdir(_sub_dir)) != nullptr) {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
            continue;
        entry_name_list.push_back(entry_name);
    }
    std::sort(entry_name_list.begin(), entry_name_list.end());

    _wds_shards.reserve(entry_name_list.size());
    // Create n such std-streams for n paths
     for (auto& path : entry_name_list)
        _wds_shards.emplace_back(StdFileStream::Open(_folder_path + path));

    // if (!_meta_data_reader) {
        // collecting and filtering the index files
        std::vector<SampleDescription> unfiltered_samples;
        std::vector<ComponentDescription> unfiltered_components;

        for (unsigned wds_shard_index = 0; wds_shard_index < entry_name_list.size(); ++wds_shard_index) {
            unfiltered_samples.resize(0);
            unfiltered_components.resize(0);
            if (_index_paths.size() == 0)
                parse_tar_files(unfiltered_samples, unfiltered_components, _wds_shards[wds_shard_index]);
            // else TODO:Swetha
                // parse_index_files(unfiltered_samples, unfiltered_components, _folder_path + entry_name_list[wds_shard_index]);

            // After parsing add the contents to the map
            for (auto& sample : unfiltered_samples) {
                for (auto& component : sample.components) {
                     if (webdataset_record_reader_from_components(component, wds_shard_index) != Reader::Status::OK)
                                WRN("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] WebDataset File reader cannot access the storage at " + _folder_path + component.filename);
                }
            }
        }

    // }
    // for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
    //     std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
    //     _folder_path = subfolder_path;
    //     if (webdataset_record_reader_from_components() != Reader::Status::OK)
    //         WRN("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] File reader cannot access the storage at " + _folder_path);
    // }

    if (!_file_names.empty())
        LOG("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path)
    closedir(_sub_dir);
    return ret;
}


Reader::Status WebDatasetSourceReader::webdataset_record_reader_from_components(ComponentDescription component, unsigned wds_shard_index) {
    auto ret = Reader::Status::OK;
    if (component.ext == "jpg") {
        // Update file path and size
        std::string file_path = _folder_path;
        file_path.append("/");
        file_path.append(component.filename);
        _file_names.push_back(file_path);
        _file_size.insert(std::pair<std::string, unsigned int>(file_path, component.size));
        _file_wds_shard_idx_mapping.insert(std::pair<std::string, unsigned int>(file_path, wds_shard_index));
        _file_offset.insert(std::pair<std::string, off_t>(file_path, component.offset));
    } // Case for jpg's.
    return ret; // Return OK status
}

Reader::Status WebDatasetSourceReader::read_web_dataset_at_offset(unsigned char *buff, std::string file_name, uint file_size, uint offset, uint wds_shard_index) {
    auto ret = Reader::Status::OK;
    auto& current_tar_file_stream = _wds_shards[wds_shard_index];
    current_tar_file_stream->SeekRead(offset);
    // Prepare to read data
    // std::vector<char> cls_data(file_size);
    current_tar_file_stream->Read(buff, file_size);
    return ret; // Return OK status
}
