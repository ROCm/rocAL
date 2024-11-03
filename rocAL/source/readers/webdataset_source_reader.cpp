
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

#include "readers/webdataset_source_reader.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cassert>

using namespace std;

#define BLOCKSIZE 10240

constexpr int create_version_number(int major, int minor, int patch = 0) {
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
    int ret = 0; // Default initialization
    if (_shard_size == -1) {
        if (_loop) return largest_shard_size_without_padding();                   // When shard_size is set to -1, The shard_size variable is not used
        int size = std::max(largest_shard_size_without_padding(), _batch_size);  // Return the size of the largest shard amongst all the shard's size
        ret = (size - _read_counter);
        // Formula used to calculate - [_last_batch_padded_size = _batch_size - (_shard_size % _batch_size) ]
        // Since "size" doesnt involve padding - we add the count of padded samples to the number of remaining elements
        // which equals to the shard size with padding
        if (_sharding_info.last_batch_policy == RocalBatchPolicy::PARTIAL || _sharding_info.last_batch_policy == RocalBatchPolicy::FILL) {
            ret += _last_batch_padded_size;
        } else if (_sharding_info.last_batch_policy == RocalBatchPolicy::DROP &&
                   _pad_last_batch_repeated == true) { // When pad_last_batch_repeated is False - Enough
                                                      // number of samples would not be present in the last batch - hence
                                                      // dropped by condition handled in the loader
            ret -= _batch_size;
        }
    } else if (_shard_size > 0) {
        auto shard_size_with_padding =
            _shard_size + (_batch_size - (_shard_size % _batch_size));
        if (_loop)
            return shard_size_with_padding;
        int size = std::max(shard_size_with_padding, _batch_size);
        ret = (size - _read_counter);
        if (_sharding_info.last_batch_policy == RocalBatchPolicy::DROP) // The shard size is padded at the beginning of the condition, hence dropping the last batch
            ret -= _batch_size;
    }
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
    _batch_size = desc.get_batch_size();
    _loop = desc.loop();
    _meta_data_reader = desc.meta_data_reader();
    _sharding_info = desc.get_sharding_info();
    _pad_last_batch_repeated = _sharding_info.pad_last_batch_repeated;
    _stick_to_shard = _sharding_info.stick_to_shard;
    _shard_size = _sharding_info.shard_size;
    _shuffle = desc.shuffle();
    ret = folder_reading();
    _curr_file_idx = _shard_start_idx_vector[_shard_id]; // shard's start_idx would vary for every shard in the vector
    // shuffle dataset if set
    if (ret == Reader::Status::OK && _shuffle)
        if (ret == Reader::Status::OK && _shuffle)
        std::random_shuffle(_file_names.begin() + _shard_start_idx_vector[_shard_id],
                            _file_names.begin() + _shard_end_idx_vector[_shard_id]);

    return ret;
}

void WebDatasetSourceReader::increment_curr_file_idx() {
    // Should work for both pad_last_batch = True (or) False
    auto shard_start_idx = _shard_start_idx_vector[_shard_id];
    if (_stick_to_shard == false) {
        _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
    } else {
        if (_curr_file_idx >= shard_start_idx &&
            _curr_file_idx < shard_start_idx + actual_shard_size_without_padding() - 1) // checking if current-element lies within the shard size [begin_idx, last_idx -1]
            _curr_file_idx = (_curr_file_idx + 1);
        else
            _curr_file_idx = shard_start_idx;
    }
}

void WebDatasetSourceReader::incremenet_read_ptr() {
    _read_counter++;
    increment_curr_file_idx();
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
        std::random_shuffle(_file_names.begin() + _shard_start_idx_vector[_shard_id],
                            _file_names.begin() + _shard_end_idx_vector[_shard_id]);
    _read_counter = 0;
    _curr_file_idx = 0;
}

void WebDatasetSourceReader::parse_sample_description(
    std::vector<SampleDescription> &samples_container,
    std::vector<ComponentDescription> &components_container,
    std::ifstream &index_file, const std::string &index_path, int64_t line,
    int index_version) {
    samples_container.emplace_back();
    samples_container.back().components = VectorView<ComponentDescription>(
        components_container, components_container.size());
    samples_container.back().line_number = line;

    // Getting the components data
    std::string components_metadata;
    std::getline(index_file, components_metadata);
    std::stringstream components_stream(components_metadata);

    // Reading consecutive components
    ComponentDescription component;
    while (components_stream >> component.ext) {

        if (index_version == create_version_number(1, 2)) {
            if (!(components_stream >> component.offset >> component.size >>
                  component.filename)) {
                THROW("Could not find all necessary component parameters "
                      "(offset, size or filename). Every record in the index "
                      "file should look like: `<ext> <offset> <size> "
                      "<filename>`.");
            }
        } else {
            if (!(components_stream >> component.offset >> component.size))
                THROW("Could not find all necessary component parameters "
                      "(offset or size). Every record in the index file should "
                      "look like: `<ext> <offset> <size>`");
        }

        if (component.filename.empty()) // Use line number as file number
            component.filename = std::to_string(line);
        else {
            // Find the position of the last period
            auto last_period_pos = component.filename.find_last_of('.');
            // If a period is found, truncate everything after it
            if (last_period_pos != std::string::npos) {
                component.filename.erase(last_period_pos);
            }
        }

        if (!(component.offset % kBlockSize == 0))
            THROW("tar offset is not a multiple of tar block size kBlockSize, "
                  "perhaps the size value is exported before offset?");

        components_container.emplace_back(std::move(component));
        samples_container.back().components.num++;
    }

    if ((!samples_container.back().components.num))
        THROW("No extensions provided for the sample");
}

inline int parse_index_version(const string& idx_version_in_str) {
  const char *c_string_ptr = idx_version_in_str.c_str();
  assert(*c_string_ptr == 'v');
  c_string_ptr++;
  auto major = atoi(c_string_ptr);
  c_string_ptr = strchr(c_string_ptr, '.');
  assert(c_string_ptr);
  c_string_ptr++;
  auto minor = atoi(c_string_ptr);
  return create_version_number(major, minor);
}

void WebDatasetSourceReader::parse_index_files(
    std::vector<SampleDescription> &samples_container,
    std::vector<ComponentDescription> &components_container,
    const std::string &index_path) {
    std::ifstream index_file(index_path);
    std::string global_meta;
    getline(index_file, global_meta);
    std::stringstream global_meta_stream(global_meta);
    std::string index_version_str;
    if (!(global_meta_stream >> index_version_str))
        THROW("Unsupported version of the index file")

    int index_version = parse_index_version(index_version_str);

    int64_t sample_desc_num_signed;
    if (!(global_meta_stream >> sample_desc_num_signed))
        THROW("no sample count found")
    if (!(sample_desc_num_signed > 0))
        THROW("sample count must be positive")

    const size_t sample_desc_num = sample_desc_num_signed;
    samples_container.reserve(samples_container.size() + sample_desc_num);
    for (size_t sample_index = 0; sample_index < sample_desc_num;
         sample_index++) {
        parse_sample_description(samples_container, components_container, index_file,
                        index_path, sample_index + 1, index_version);
    }
}


void WebDatasetSourceReader::parse_tar_files(std::vector<SampleDescription>& samples_vector,
                                              std::vector<ComponentDescription>& components_vector,
                                              std::unique_ptr<FileIOStream>& tar_file) {
    TarArchive tar_archive(std::move(tar_file));

    std::string last_filename;
    for (; !tar_archive.at_end_of_archive(); tar_archive.advance_to_next_file_in_tar()) {
        if (tar_archive.get_current_file_type() == TarArchive::ENTRY_FILE) {
        std::tie(last_filename, std::ignore) = split_name(tar_archive.get_current_file_name());
        break;
        }
    }
    size_t last_components_size = components_vector.size();
    for (; !tar_archive.at_end_of_archive(); tar_archive.advance_to_next_file_in_tar()) {
        if (tar_archive.get_current_file_type() != TarArchive::ENTRY_FILE) {
        continue;
        }

    std::string basename, ext;
    std::tie(basename, ext) = split_name(tar_archive.get_current_file_name());
    if (basename.empty()) {
      continue;
    }

    if (basename != last_filename) {
      samples_vector.emplace_back();
      samples_vector.back().components = VectorView<ComponentDescription>(components_vector, last_components_size, components_vector.size() - last_components_size);
      last_filename = basename;
      last_components_size = components_vector.size();
    }

    components_vector.emplace_back();
    components_vector.back().size = tar_archive.get_current_file_size();
    components_vector.back().offset = tar_archive.get_current_archive_offset() + tar_archive.get_current_header_size();
    components_vector.back().ext = std::move(ext);
    auto _last_id = basename;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        _last_id.erase(0, last_slash_idx + 1);
    }
    components_vector.back().filename = _last_id;

  }
    samples_vector.emplace_back();
    samples_vector.back().components = VectorView<ComponentDescription>(components_vector, last_components_size, components_vector.size() - last_components_size);

    tar_file = tar_archive.release_file_stream();

}

Reader::Status WebDatasetSourceReader::folder_reading() {
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
            _wds_shards.emplace_back(FileIOStream::open(_path + path));
    }
    else {
            _folder_path = _index_paths;
            if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
                THROW("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);
            _full_path = _folder_path;
            while ((_entity = readdir(_sub_dir)) != nullptr) {
                std::string entry_name(_entity->d_name);
                if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                    continue;
                _index_name_list.push_back(entry_name);
            }
            std::sort(_index_name_list.begin(), _index_name_list.end());
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
                _wds_shards.emplace_back(FileIOStream::open(_path + path));
    }

        std::vector<SampleDescription> unfiltered_samples;
        std::vector<ComponentDescription> unfiltered_components;

        for (unsigned wds_shard_index = 0; wds_shard_index < entry_name_list.size(); ++wds_shard_index) {
            unfiltered_samples.resize(0);
            unfiltered_components.resize(0);
            if (_index_paths.size() == 0)
                parse_tar_files(unfiltered_samples, unfiltered_components, _wds_shards[wds_shard_index]);
            else
                parse_index_files(unfiltered_samples, unfiltered_components, _folder_path + _index_name_list[wds_shard_index]);
            
            // After parsing add the contents to the map
            for (auto& sample : unfiltered_samples) {
                for (auto& component : sample.components) {
                    if (!_meta_data_reader || _meta_data_reader->exists(component.filename)) {
                        if (webdataset_record_reader_from_components(component, wds_shard_index) != Reader::Status::OK)
                                    WRN("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] WebDataset File reader cannot access the storage at " + _folder_path + component.filename);
                    }
                }
            }
   
    }
    auto dataset_size = _file_count_all_shards;
    // Pad the _file_names with last element of the shard in the vector when _pad_last_batch_repeated is True
    if (_shard_size > 0)
        _padded_samples = _shard_size % _batch_size;
    else
        _padded_samples = largest_shard_size_without_padding() % _batch_size;
    if (_padded_samples != 0)
        _last_batch_padded_size = _batch_size - _padded_samples;

    if (_pad_last_batch_repeated == true) { 
        // pad the last sample when the dataset_size is not divisible by
        // the number of shard's (or) when the shard's size is not
        // divisible by the batch size making each shard having equal
        // number of samples
        uint32_t total_padded_samples = 0; // initialize the total_padded_samples to 0
        for (uint shard_id = 0; shard_id < _shard_count; shard_id++) {
            uint start_idx = (dataset_size * shard_id) / _shard_count;
            uint actual_shard_size_without_padding = std::floor((shard_id + 1) * dataset_size / _shard_count) - floor(shard_id * dataset_size / _shard_count);
            uint largest_shard_size = std::ceil(dataset_size * 1.0 / _shard_count);

            auto start = _file_names.begin() + start_idx + total_padded_samples;
            auto end = start + actual_shard_size_without_padding;
            if (largest_shard_size % _batch_size) {
                size_t num_padded_samples = 0;
                num_padded_samples = (largest_shard_size - actual_shard_size_without_padding) + _batch_size - (largest_shard_size % _batch_size);
                _file_count_all_shards += _num_padded_samples;
                _file_names.insert(end, num_padded_samples, _file_names[start_idx + actual_shard_size_without_padding + total_padded_samples - 1]);
                total_padded_samples += num_padded_samples;
            }
        }
    }
    if (!_file_names.empty())
        LOG("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path)
    compute_start_and_end_idx_of_all_shards();
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
        _file_count_all_shards++;
    } // Case for jpg's. - add for more extensions when encoutered
    return ret;
}

Reader::Status WebDatasetSourceReader::read_web_dataset_at_offset(unsigned char *buff, std::string file_name, uint file_size, uint offset, uint wds_shard_index) {
    auto ret = Reader::Status::OK;
    auto& current_tar_file_stream = _wds_shards[wds_shard_index];
    current_tar_file_stream->set_read_position(offset);
    current_tar_file_stream->read_into_buffer(buff, file_size);
    return ret;
}

size_t WebDatasetSourceReader::get_dataset_size() {
    return _file_count_all_shards;
}

size_t WebDatasetSourceReader::actual_shard_size_without_padding() {
    return std::floor((_shard_id + 1) * get_dataset_size() / _shard_count) - floor(_shard_id * get_dataset_size() / _shard_count);
}

size_t WebDatasetSourceReader::largest_shard_size_without_padding() {
  return std::ceil(get_dataset_size() * 1.0 / _shard_count);
}

void WebDatasetSourceReader::increment_shard_id() {
    _shard_id = (_shard_id + 1) % _shard_count;
}

