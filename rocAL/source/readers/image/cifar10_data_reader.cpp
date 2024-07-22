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

#include <cassert>
#include "pipeline/commons.h"
#include <cstring>
#include <algorithm>
#include <math.h>
#include "readers/image/cifar10_data_reader.h"
#include "readers/file_source_reader.h"
#include "pipeline/filesystem.h"

CIFAR10DataReader::CIFAR10DataReader() {
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
    _total_file_size = 0;
    _last_file_idx = 0;
    _file_count_all_shards = 0;
}

unsigned CIFAR10DataReader::count_items() {
    int ret = 0; // Default initialization
    if (_shard_size == -1) {
        if (_loop) return largest_shard_size_without_padding();                   // When shard_size is set to -1, The shard_size variable is not used
        int size = std::max(largest_shard_size_without_padding(), _batch_count);  // Return the size of the largest shard amongst all the shard's size
        ret = (size - _read_counter);
        // Formula used to calculate - [_last_batch_padded_size = _batch_count - (_shard_size % _batch_count) ]
        // Since "size" doesnt involve padding - we add the count of padded samples to the number of remaining elements
        // which equals to the shard size with padding
        if (_last_batch_info.first == RocalBatchPolicy::PARTIAL || _last_batch_info.first == RocalBatchPolicy::FILL) {
            ret += _last_batch_padded_size;
        } else if (_last_batch_info.first == RocalBatchPolicy::DROP &&
                   _last_batch_info.second == true) { // When pad_last_batch_repeated is False - Enough
                                                      // number of samples would not be present in the last batch - hence
                                                      // dropped by condition handled in the loader
            ret -= _batch_count;
        }
    } else if (_shard_size > 0) {
        auto shard_size_with_padding =
            _shard_size + (_batch_count - (_shard_size % _batch_count));
        if (_loop)
            return shard_size_with_padding;
        int size = std::max(shard_size_with_padding, _batch_count);
        ret = (size - _read_counter);
        if (_last_batch_info.first == RocalBatchPolicy::DROP) // The shard size is padded at the beginning of the condition, hence dropping the last batch
            ret -= _batch_count;
    }
    return ((ret < 0) ? 0 : ret);
}

Reader::Status CIFAR10DataReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _folder_path = desc.path();
    _batch_count = desc.get_batch_size();
    _loop = desc.loop();
    _file_name_prefix = desc.file_prefix();
    _pad_last_batch_repeated = _last_batch_info.second;
    _stick_to_shard = desc.get_stick_to_shard();
    _shard_size = desc.get_shard_size();
    ret = subfolder_reading();
    _curr_file_idx = get_start_idx(); // shard's start_idx would vary for every shard in the vector
    return ret;

}

void CIFAR10DataReader::increment_curr_file_idx() {
    // Should work for both pad_last_batch = True (or) False
    auto shard_start_idx = get_start_idx();
    if (_stick_to_shard == false) {
        _curr_file_idx = (_curr_file_idx + 1) % _all_shard_file_names_padded.size();
    } else {
        if (_curr_file_idx >= shard_start_idx &&
            _curr_file_idx < shard_start_idx + actual_shard_size_without_padding() - 1) // checking if current-element lies within the shard size [begin_idx, last_idx -1]
            _curr_file_idx = (_curr_file_idx + 1);
        else
            _curr_file_idx = shard_start_idx;
    }
}

void CIFAR10DataReader::incremenet_read_ptr() {
    _read_counter++;
    increment_curr_file_idx();
}

size_t CIFAR10DataReader::open() {
    auto file_path = _file_names[_curr_file_idx];  // Get next file name
    auto file_offset = _file_offsets[_curr_file_idx];
    _last_file_idx = _file_idx[_curr_file_idx];
    incremenet_read_ptr();
    // update _last_id for the next record
    _last_id = file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        _last_id.erase(0, last_slash_idx + 1);
    }
    // add file_idx to last_id so the loader knows the index within the same master file
    _last_id.append("_");
    _last_id.append(std::to_string(_last_file_idx));
    // compare the file_name with the last one opened
    if (file_path.compare(_last_file_name) != 0) {
        if (_current_fPtr) {
            fclose(_current_fPtr);
            _current_fPtr = nullptr;
        }
        _current_fPtr = fopen(file_path.c_str(), "rb");  // Open the file,
        _last_file_name = file_path;
        fseek(_current_fPtr, 0, SEEK_END);  // Take the file read pointer to the end
        _total_file_size = ftell(_current_fPtr);
        fseek(_current_fPtr, 0, SEEK_SET);  // Take the file read pointer to the beginning
    }

    if (!_current_fPtr)  // Check if it is ready for reading
        return 0;

    fseek(_current_fPtr, file_offset, SEEK_END);  // Take the file read pointer to the end

    _current_file_size = ftell(_current_fPtr);  // Check how many bytes are there between and the current read pointer position (end of the file)

    if (_current_file_size < _raw_file_size)  // not enough data in the file to read
    {                                         // If file is empty continue
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        return 0;
    }

    fseek(_current_fPtr, file_offset + 1, SEEK_SET);  // Take the file pointer back to the fileoffset + 1 extra byte for label

    return (_raw_file_size - 1);
}

size_t CIFAR10DataReader::read_data(unsigned char* buf, size_t read_size) {
    if (!_current_fPtr)
        return 0;

    // Requested read size bigger than the raw file size? just read as many bytes as the raw file size
    read_size = (read_size > (_raw_file_size - 1)) ? _raw_file_size - 1 : read_size;

    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
    return actual_read_size;
}

int CIFAR10DataReader::close() {
    return release();
}

CIFAR10DataReader::~CIFAR10DataReader() {
    if (_current_fPtr) {
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
    }
}

int CIFAR10DataReader::release() {
    // do not need to close file here since data is read from the same file continuously
    return 0;
}

void CIFAR10DataReader::reset() {
    if (_stick_to_shard == false)  // Pick elements from the next shard - hence increment shard_id
        increment_shard_id();      // Should work for both single and multiple shards

    _read_counter = 0;

    if (_last_batch_info.first == RocalBatchPolicy::DROP) {  // Skipping the dropped batch in next epoch
        for (uint i = 0; i < _batch_count; i++)
            increment_curr_file_idx();
    }
}

Reader::Status CIFAR10DataReader::subfolder_reading() {
    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("CIFAR10DataReader ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;

    while ((_entity = readdir(_sub_dir)) != nullptr) {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
        LOG("CIFAR10DataReader  Got entry name " + entry_name)
    }
    std::sort(entry_name_list.begin(), entry_name_list.end());
    std::string subfolder_path = _full_path + "/" + entry_name_list[0];
    filesys::path pathObj(subfolder_path);
    auto ret = Reader::Status::OK;
    if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
        ret = open_folder();
    } else if (filesys::exists(pathObj) && filesys::is_directory(pathObj)) {
        for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
            std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
            _folder_path = subfolder_path;
            if (open_folder() != Reader::Status::OK)
                WRN("CIFAR10DataReader: File reader cannot access the storage at " + _folder_path);
        }
    }
    if (!_file_names.empty())
        LOG("CIFAR10DataReader  Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path)

    auto dataset_size = _file_count_all_shards;
    // Pad the _file_names with last element of the shard in the vector when _pad_last_batch_repeated is True
    if (_shard_size > 0)
        _padded_samples = _shard_size % _batch_count;
    else
        _padded_samples = largest_shard_size_without_padding() % _batch_count;
    if (_padded_samples > 0)
        _last_batch_padded_size = _batch_count - _padded_samples;

    if (_pad_last_batch_repeated == true) { 
        // pad the last sample when the dataset_size is not divisible by
        // the number of shard's (or) when the shard's size is not
        // divisible by the batch size making each shard having equal
        // number of samples
        for (uint shard_id = 0; shard_id < _shard_count; shard_id++) {
            uint start_idx = (dataset_size * shard_id) / _shard_count;
            uint shard_size_without_padding = std::floor((shard_id + 1) * dataset_size / _shard_count) - floor(shard_id * dataset_size / _shard_count);
            uint shard_size_with_padding = std::ceil(dataset_size * 1.0 / _shard_count);
            auto start = _file_names.begin() + start_idx;
            auto end = _file_names.begin() + start_idx + shard_size_without_padding;
            auto start_offset = _file_offsets.begin() + start_idx;
            auto end_offset = _file_offsets.begin() + start_idx + shard_size_without_padding;
            auto start_file_idx = _file_idx.begin() + start_idx;
            auto end_file_idx = _file_idx.begin() + start_idx + shard_size_without_padding;
            if (start != end && start <= _file_names.end() &&
                end <= _file_names.end()) {
                _all_shard_file_names_padded.insert(_all_shard_file_names_padded.end(), start, end);
                _all_shard_file_offsets.insert(_all_shard_file_offsets.end(), start_offset, end_offset);
                _all_shard_file_idxs.insert(_all_shard_file_idxs.end(), start_file_idx, end_file_idx);
            }
            if (shard_size_with_padding % _batch_count) {
                _num_padded_samples = (shard_size_with_padding - shard_size_without_padding) + _batch_count - (shard_size_with_padding % _batch_count);
                _file_count_all_shards += _num_padded_samples;
                _all_shard_file_names_padded.insert(_all_shard_file_names_padded.end(), _num_padded_samples, _all_shard_file_names_padded.back());
                _all_shard_file_offsets.insert(_all_shard_file_offsets.end(), _num_padded_samples, _all_shard_file_offsets.back());
                _all_shard_file_idxs.insert(_all_shard_file_idxs.end(), _num_padded_samples, _all_shard_file_idxs.back());
            }
        }
    } else {
        _all_shard_file_names_padded = _file_names;
        _all_shard_file_offsets = _file_offsets;
        _all_shard_file_idxs = _file_idx;
    }

    _last_file_name = _all_shard_file_names_padded[_all_shard_file_names_padded.size() - 1];

    closedir(_sub_dir);
    return ret;
}

Reader::Status CIFAR10DataReader::open_folder() {
    if ((_src_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("CIFAR10DataReader ERROR: Failed opening the directory at " + _folder_path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        if (_entity->d_type != DT_REG)
            continue;
        std::string file_path = _folder_path;
        // check if the filename has the _file_name_prefix
        std::string data_file_name = std::string(_entity->d_name);
        if (data_file_name.find(_file_name_prefix) != std::string::npos) {
            file_path.append("/");
            file_path.append(_entity->d_name);
            FILE* fp = fopen(file_path.c_str(), "rb");  // Open the file,
            fseek(fp, 0, SEEK_END);                     // Take the file read pointer to the end
            size_t total_file_size = ftell(fp);
            size_t num_of_raw_files = _raw_file_size ? total_file_size / _raw_file_size : 0;
            unsigned file_offset = 0;
            for (unsigned i = 0; i < num_of_raw_files; i++) {
                _file_names.push_back(file_path);
                _file_offsets.push_back(file_offset);
                _file_idx.push_back(i);
                _file_count_all_shards++;
                file_offset += _raw_file_size;
            }

            fclose(fp);
        }
    }
    if (_file_names.empty())
        WRN("CIFAR10DataReader:: Did not load any file from " + _folder_path)

    closedir(_src_dir);
    return Reader::Status::OK;
}

size_t CIFAR10DataReader::last_batch_padded_size() {
    return _last_batch_padded_size;
}

size_t CIFAR10DataReader::get_start_idx() {
    _shard_start_idx = (get_dataset_size() * _shard_id) / _shard_count;
    return _shard_start_idx;
}

size_t CIFAR10DataReader::get_dataset_size() {
    return _file_count_all_shards;
}


size_t CIFAR10DataReader::actual_shard_size_without_padding() {
    return std::floor((_shard_id + 1) * get_dataset_size() / _shard_count) - floor(_shard_id * get_dataset_size() / _shard_count);
}

size_t CIFAR10DataReader::largest_shard_size_without_padding() {
  return std::ceil(get_dataset_size() * 1.0 / _shard_count);
}

void CIFAR10DataReader::increment_shard_id() {
    _shard_id = (_shard_id + 1) % _shard_count;
}
