
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
#include <cassert>

using namespace std;

#define BLOCKSIZE 10240

// gets single int that can be represented as int value
constexpr int MakeVersionNumber(int major, int minor, int patch = 0) {
  if (major < 0 || minor < 0 || patch < 0) {
    return -1;
  }
  return major*1000 + minor*10 + patch;
}

inline std::tuple<std::string, std::string> split_name(const std::string& file_path) {
  size_t dot_pos = file_path.find('.', file_path.rfind('/') + 1);
  return {file_path.substr(0, dot_pos), file_path.substr(dot_pos + 1)};
}

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
    auto ret = Reader::Status::OK;
    _file_id = 0;
    _folder_path = desc.path();
    _path = desc.path();
    _index_paths = desc.index_path();
    _wds_shards.reserve(_path.size());
    _feature_key_map = desc.feature_key_map();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _loop = desc.loop();
    _meta_data_reader = desc.meta_data_reader();
    _shuffle = desc.shuffle();
    ret = folder_reading();
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

void WebDatasetSourceReader::ParseSampleDesc(std::vector<SampleDescription>& samples_container,
                            std::vector<ComponentDescription>& components_container,
                            std::ifstream& index_file, const std::string& index_path, int64_t line,
                            int index_version) {
  // Preparing the SampleDescription
  samples_container.emplace_back();
  samples_container.back().components = VectorView<ComponentDescription>(components_container, components_container.size());
  samples_container.back().line_number = line;

  // Getting the components data
  std::string components_metadata;
  std::getline(index_file, components_metadata);
  std::stringstream components_stream(components_metadata);

  // Reading consecutive components
  ComponentDescription component;
  while (components_stream >> component.ext) {
    if (index_version == MakeVersionNumber(1, 2)) {
      if(components_stream >> component.offset >> component.size >> component.filename)
      std::cerr << "Could not find all necessary component parameters (offset, size or filename). Every record in the index file should look like: `<ext> <offset> <size> <filename>`.";
        //   IndexFileErrMsg(index_path, line, "Could not find all necessary component parameters (offset, size or filename). Every record in the index file should look like: `<ext> <offset> <size> <filename>`.");
        std::string basename, ext;
        std::tie(basename, ext) = split_name(component.filename);
        auto _last_id = basename;
        auto last_slash_idx = _last_id.find_last_of("\\/");
        if (std::string::npos != last_slash_idx) {
            _last_id.erase(0, last_slash_idx + 1);
        }
    } else {
      if(components_stream >> component.offset >> component.size)
            std::cerr << "Could not find all necessary component parameters (offset or size). Every record in the index file should look like: `<ext> <offset> <size>`";
        // IndexFileErrMsg(index_path, line, "Could not find all necessary component parameters (offset or size). Every record in the index file should look like: `<ext> <offset> <size>`.");
    }
    if(component.offset % kBlockSize == 0)
        std::cerr << "tar offset is not a multiple of tar block size kBlockSize, perhaps the size value is exported before offset?";
        // IndexFileErrMsg(index_path, line, "tar offset is not a multiple of tar block size kBlockSize, perhaps the size value is exported before offset?");
    if (component.filename.empty()) // Use line number as file number
        component.filename = std::to_string(line);
    components_container.emplace_back(std::move(component));
    samples_container.back().components.num++;
  }

  // Finishing up the SampleDescription
  if(samples_container.back().components.num)
        std::cerr << "\n no extensions provided for the sample";
        // IndexFileErrMsg(index_path, line, "no extensions provided for the sample");
}

inline int ParseIndexVersion(const string& version_str) {
  const char *s = version_str.c_str();
  assert(*s == 'v');
  s++;
  int major = atoi(s);
  s = strchr(s, '.');
  assert(s);
  s++;
  int minor = atoi(s);
  return MakeVersionNumber(major, minor);
}

void WebDatasetSourceReader::parse_index_files(std::vector<SampleDescription>& samples_container,
                           std::vector<ComponentDescription>& components_container,
                           const std::string& index_path) {
  std::ifstream index_file(index_path);
    std::cerr << "\n index_path :: " << index_path;
  // Index Checking
  std::string global_meta;
  getline(index_file, global_meta);
  std::stringstream global_meta_stream(global_meta);
  std::string index_version_str;
  if (!(global_meta_stream >> index_version_str))
    THROW("Unsupported version of the index file ");

  int index_version = ParseIndexVersion(index_version_str);

  // Getting the number of samples in the index file
  int64_t sample_desc_num_signed;
  if(!(global_meta_stream >> sample_desc_num_signed))
    THROW("no sample count found");
    // IndexFileErrMsg(index_path, 0, "no sample count found");
    std::cerr << "\n sample_desc_num_signed" << sample_desc_num_signed;
  if(!(sample_desc_num_signed > 0))
    THROW("sample count must be positive");
    // IndexFileErrMsg(index_path, 0, "sample count must be positive");

  const size_t sample_desc_num = sample_desc_num_signed;
  samples_container.reserve(samples_container.size() + sample_desc_num);
  for (size_t sample_index = 0; sample_index < sample_desc_num; sample_index++) {
    ParseSampleDesc(samples_container, components_container, index_file, index_path,
                    sample_index + 1, index_version);
  }
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
    auto _last_id = basename;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        _last_id.erase(0, last_slash_idx + 1);
    }
    components_container.back().filename = _last_id;

  }
    samples_container.emplace_back();
    samples_container.back().components = VectorView<ComponentDescription>(components_container, last_components_size, components_container.size() - last_components_size);

    tar_file = tar_archive.Release();

}

Reader::Status WebDatasetSourceReader::folder_reading() {
    std::cerr << "\n WebDatasetSourceReader::folder_reading ";
    auto ret = Reader::Status::OK;
    std::string _full_path;
    std::vector<std::string> entry_name_list;
    if (_index_paths.size() == 0) { 
        _folder_path = _path;
            if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

        _full_path = _folder_path;

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
            _wds_shards.emplace_back(StdFileStream::Open(_path + path));
    }
    else {
            // index file path
            _folder_path = _index_paths;
            if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
                THROW("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);
        
            _full_path = _folder_path;

            while ((_entity = readdir(_sub_dir)) != nullptr) {
                std::string entry_name(_entity->d_name);
                if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                    continue;
                index_name_list.push_back(entry_name);
            }
            std::sort(index_name_list.begin(), index_name_list.end());

            // tar file path

            if ((_sub_dir = opendir(_path.c_str())) == nullptr)
                THROW("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _path);

            std::string _full_path = _path;

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
                _wds_shards.emplace_back(StdFileStream::Open(_path + path));
    }


    // if (!_meta_data_reader) {
        // collecting and filtering the index files
        std::vector<SampleDescription> unfiltered_samples;
        std::vector<ComponentDescription> unfiltered_components;

        for (unsigned wds_shard_index = 0; wds_shard_index < entry_name_list.size(); ++wds_shard_index) {
            unfiltered_samples.resize(0);
            unfiltered_components.resize(0);
            if (_index_paths.size() == 0)
                parse_tar_files(unfiltered_samples, unfiltered_components, _wds_shards[wds_shard_index]);
            else
                parse_index_files(unfiltered_samples, unfiltered_components, _folder_path + index_name_list[wds_shard_index]);

            // After parsing add the contents to the map
            for (auto& sample : unfiltered_samples) {
                for (auto& component : sample.components) {
                     if (webdataset_record_reader_from_components(component, wds_shard_index) != Reader::Status::OK)
                                WRN("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] WebDataset File reader cannot access the storage at " + _folder_path + component.filename);
                }
            }
        }

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
