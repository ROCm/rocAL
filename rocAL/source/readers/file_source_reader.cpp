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
#include <algorithm>
#include <cstring>
#include <math.h>
#include "pipeline/commons.h"
#include "readers/file_source_reader.h"
#include "pipeline/filesystem.h"

FileSourceReader::FileSourceReader() {
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
    _file_id = 0;
    _shuffle = false;
    _file_count_all_shards = 0;
}

unsigned FileSourceReader::count_items() {
    int ret = 0;
    if (_shard_size == -1) {                                     // When shard_size is set to -1, The shard_size variable is not used
        if (_loop) return largest_shard_size_without_padding();  // Return the size of the largest shard amongst all the shard's size
        int size = std::max(largest_shard_size_without_padding(), _batch_size);
        ret = (size - _read_counter);
        // Formula used to calculate - [_last_batch_padded_size = _batch_size - (_shard_size % _batch_size) ]
        // Since "size" doesnt involve padding - we add the count of padded samples to the number of remaining elements
        // which equals to the shard size with padding
        if (_last_batch_info.last_batch_policy == RocalBatchPolicy::PARTIAL || _last_batch_info.last_batch_policy == RocalBatchPolicy::FILL) {
            ret += _last_batch_padded_size;
        } else if (_last_batch_info.last_batch_policy == RocalBatchPolicy::DROP &&
                   _last_batch_info.pad_last_batch_repeated == true) {  // When pad_last_batch_repeated is False - Enough
                                                       // number of samples would not be present in the last batch - hence
                                                       // dropped by condition handled in the loader
            ret -= _batch_size;
        }
    } else if (_shard_size > 0) {
        auto largest_shard_size_with_padding =
            _shard_size + (_batch_size - (_shard_size % _batch_size));  // The shard size used here is padded
        if (_loop)
            return largest_shard_size_with_padding;
        int size = std::max(largest_shard_size_with_padding, _batch_size);
        ret = (size - _read_counter);
        if (_last_batch_info.last_batch_policy == RocalBatchPolicy::DROP)  // The shard size is padded at the beginning of the condition, hence dropping the last batch
            ret -= _batch_size;
    }
    return ((ret < 0) ? 0 : ret);
}

Reader::Status FileSourceReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _file_id = 0;
    _folder_path = desc.path();
    _file_list_path = desc.file_list_path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_size = desc.get_batch_size();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    _meta_data_reader = desc.meta_data_reader();
    _last_batch_info = desc.get_sharding_info();
    _pad_last_batch_repeated = _last_batch_info.pad_last_batch_repeated;
    _stick_to_shard = _last_batch_info.stick_to_shard;
    _shard_size = _last_batch_info.shard_size;
    ret = subfolder_reading();
    // shuffle dataset if set
    if (ret == Reader::Status::OK && _shuffle)
        std::random_shuffle(_file_names.begin() + _shard_start_idx_vector[_shard_id],
                            _file_names.begin() + _shard_end_idx_vector[_shard_id]);

    return ret;
}

void FileSourceReader::increment_curr_file_idx() {
    // The condition satisfies for both pad_last_batch = True (or) False
    if (_stick_to_shard == false) {  // The elements of each shard rotate in a round-robin fashion once the elements in particular shard is exhausted
        _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
    } else {  // Stick to only elements from the current shard
        if (_curr_file_idx >= _shard_start_idx_vector[_shard_id] &&
            _curr_file_idx < _shard_end_idx_vector[_shard_id])  // checking if current-element lies within the shard size [begin_idx, last_idx -1]
            _curr_file_idx = (_curr_file_idx + 1);
        else
            _curr_file_idx = _shard_start_idx_vector[_shard_id];
    }
}

void FileSourceReader::incremenet_read_ptr() {
    _read_counter++;
    increment_curr_file_idx();
}

size_t FileSourceReader::open() {
    auto file_path = _file_names[_curr_file_idx];  // Get next file name
    incremenet_read_ptr();
    _last_file_path = _last_id = file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        _last_id.erase(0, last_slash_idx + 1);
    }

    _current_fPtr = fopen(file_path.c_str(), "rb");  // Open the file,

    if (!_current_fPtr)  // Check if it is ready for reading
        return 0;

    fseek(_current_fPtr, 0, SEEK_END);  // Take the file read pointer to the end

    _current_file_size = ftell(_current_fPtr);  // Check how many bytes are there between and the current read pointer position (end of the file)

    if (_current_file_size == 0) {  // If file is empty continue
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        return 0;
    }

    fseek(_current_fPtr, 0, SEEK_SET);  // Take the file pointer back to the start

    return _current_file_size;
}

size_t FileSourceReader::read_data(unsigned char* buf, size_t read_size) {
    if (!_current_fPtr)
        return 0;

    // Requested read size bigger than the file size? just read as many bytes as the file size
    read_size = (read_size > _current_file_size) ? _current_file_size : read_size;

    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
    return actual_read_size;
}

int FileSourceReader::close() {
    return release();
}

FileSourceReader::~FileSourceReader() {
    release();
}

int FileSourceReader::release() {
    if (!_current_fPtr)
        return 0;
    fclose(_current_fPtr);
    _current_fPtr = nullptr;
    return 0;
}

void FileSourceReader::reset() {
    if (_shuffle)
        std::random_shuffle(_file_names.begin() + _shard_start_idx_vector[_shard_id],
                            _file_names.begin() + _shard_start_idx_vector[_shard_id] + actual_shard_size_without_padding());

    if (_stick_to_shard == false)  // Pick elements from the next shard - hence increment shard_id
        increment_shard_id();      // Should work for both single and multiple shards

    _read_counter = 0;

    if (_last_batch_info.last_batch_policy == RocalBatchPolicy::DROP) {  // Skipping the dropped batch in next epoch
        for (uint i = 0; i < _batch_size; i++)
            increment_curr_file_idx();
    }
}

void FileSourceReader::increment_shard_id() {
    _shard_id = (_shard_id + 1) % _shard_count;
}

Reader::Status FileSourceReader::generate_file_names() {
    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;

    while ((_entity = readdir(_sub_dir)) != nullptr) {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
    }
    closedir(_sub_dir);
    std::sort(entry_name_list.begin(), entry_name_list.end());

    auto ret = Reader::Status::OK;
    if (!_file_list_path.empty()) {  // Reads the file paths from the file list and adds to file_names vector for decoding
        if (_meta_data_reader) {
            auto vec_rel_file_path = _meta_data_reader->get_relative_file_path();  // Get the relative file path's from meta_data_reader
            for (auto file_path : vec_rel_file_path) {
                if (filesys::path(file_path).is_relative()) {  // Only add root path if the file list contains relative file paths
                    if (!filesys::exists(_folder_path))
                        THROW("File list contains relative paths but root path doesn't exists");
                    _absolute_file_path = _folder_path + "/" + file_path;
                }
                if (filesys::is_regular_file(_absolute_file_path)) {
                    _file_names.push_back(_absolute_file_path);
                    _file_count_all_shards++;
                    incremenet_file_id();
                }
            }
            _last_file_name = _absolute_file_path;
        } else {
            std::ifstream fp(_file_list_path);
            if (fp.is_open()) {
                while (fp) {
                    std::string file_label_path;
                    std::getline(fp, file_label_path);
                    std::istringstream ss(file_label_path);
                    std::string file_path;
                    std::getline(ss, file_path, ' ');
                    if (filesys::path(file_path).is_relative()) {  // Only add root path if the file list contains relative file paths
                        if (!filesys::exists(_folder_path))
                            THROW("File list contains relative paths but root path doesn't exists");
                        file_path = _folder_path + "/" + file_path;
                    }
                    std::string file_name = file_path.substr(file_path.find_last_of("/\\") + 1);

                    if (filesys::is_regular_file(file_path)) {
                        _last_file_name = file_path;
                        _file_names.push_back(file_path);
                        _file_count_all_shards++;
                        incremenet_file_id();
                    }
                }
            }
        }
    } else {
        for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
            std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
            filesys::path pathObj(subfolder_path);
            if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
                // ignore files with unsupported extensions
                auto file_extension_idx = subfolder_path.find_last_of(".");
                if (file_extension_idx != std::string::npos) {
                    std::string file_extension = subfolder_path.substr(file_extension_idx + 1);
                    std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    if ((file_extension != "jpg") && (file_extension != "jpeg") && (file_extension != "png") && (file_extension != "ppm") && (file_extension != "bmp") && (file_extension != "pgm") && (file_extension != "tif") && (file_extension != "tiff") && (file_extension != "webp") && (file_extension != "wav"))
                        continue;
                }
                ret = open_folder();
                break;  // assume directory has only files.
            } else if (filesys::exists(pathObj) && filesys::is_directory(pathObj)) {
                _folder_path = subfolder_path;
                if (open_folder() != Reader::Status::OK)
                    WRN("FileReader ShardID [" + TOSTR(_shard_id) + "] File reader cannot access the storage at " + _folder_path);
            }
        }
    }

    if (_file_names.empty())
        ERR("FileReader ShardID [" + TOSTR(_shard_id) + "] Did not load any file from " + _folder_path)

    auto dataset_size = _file_count_all_shards;
    // Pad the _file_names with last element of the shard in the vector when _pad_last_batch_repeated is True
    _padded_samples = ((_shard_size > 0) ? _shard_size : largest_shard_size_without_padding()) % _batch_size;
    _last_batch_padded_size = (_batch_size > 1) ? (_batch_size - _padded_samples) : 0;

    if (_pad_last_batch_repeated == true) {
                                            // pad the last sample when the dataset_size is not divisible by
                                            // the number of shard's (or) when the shard's size is not
                                            // divisible by the batch size making each shard having equal
                                            // number of samples
        uint32_t total_padded_samples = 0; // initialize the total_padded_samples to 0
        for (uint32_t shard_id = 0; shard_id < _shard_count; shard_id++) {
            uint32_t start_idx = (dataset_size * shard_id) / _shard_count;
            uint32_t actual_shard_size_without_padding = std::floor((shard_id + 1) * dataset_size / _shard_count) - std::floor(shard_id * dataset_size / _shard_count);
            uint32_t largest_shard_size = std::ceil(dataset_size * 1.0 / _shard_count);
            auto start = _file_names.begin() + start_idx + total_padded_samples;
            auto end = start + actual_shard_size_without_padding;
            if (largest_shard_size % _batch_size) {
                _num_padded_samples = (largest_shard_size - actual_shard_size_without_padding) + _batch_size - (largest_shard_size % _batch_size);
                _file_count_all_shards += _num_padded_samples;
                _file_names.insert(end, _num_padded_samples, _file_names[start_idx + actual_shard_size_without_padding + total_padded_samples - 1]);
                total_padded_samples += _num_padded_samples;
            }
        }
    }

    _last_file_name = _file_names[_file_names.size() - 1];
    compute_start_and_end_idx_of_all_shards();

    return ret;
}

Reader::Status FileSourceReader::subfolder_reading() {
    auto ret = generate_file_names();
    if (!_file_names.empty())
        LOG("FileReader ShardID [" + TOSTR(_shard_id) + "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + STR(_folder_path))
    return ret;
}

Reader::Status FileSourceReader::open_folder() {
    if ((_src_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

    // Sort all the files inside the directory and then process them for sharding
    std::vector<filesys::path> files_in_directory;
    std::copy(filesys::directory_iterator(filesys::path(_folder_path)), filesys::directory_iterator(), std::back_inserter(files_in_directory));
    std::sort(files_in_directory.begin(), files_in_directory.end());
    for (const std::string file_path : files_in_directory) {
        std::string filename = file_path.substr(file_path.find_last_of("/\\") + 1);
        if (!filesys::is_regular_file(filesys::path(file_path)))
            continue;

        auto file_extension_idx = file_path.find_last_of(".");
        if (file_extension_idx != std::string::npos) {
            std::string file_extension = file_path.substr(file_extension_idx + 1);
            std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if ((file_extension != "jpg") && (file_extension != "jpeg") && (file_extension != "png") && (file_extension != "ppm") && (file_extension != "bmp") && (file_extension != "pgm") && (file_extension != "tif") && (file_extension != "tiff") && (file_extension != "webp") && (file_extension != "wav"))
                continue;
        }
        if (!_meta_data_reader || _meta_data_reader->exists(filename)) {  // Check if the file is present in metadata reader and add to file names list, to avoid issues while lookup
            _file_names.push_back(file_path);
            _last_file_name = file_path;
            _file_count_all_shards++;
            incremenet_file_id();
        } else {
            WRN("Skipping file," + filename + " as it is not present in metadata reader")
        }
    }

    if (_file_names.empty())
        ERR("FileReader ShardID [" + TOSTR(_shard_id) + "] Did not load any file from " + _folder_path)
    closedir(_src_dir);
    return Reader::Status::OK;
}

size_t FileSourceReader::get_file_shard_id() {
    if (_batch_size == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    return _file_id % _shard_count;
}

size_t FileSourceReader::last_batch_padded_size() {
    return _last_batch_padded_size;
}
std::string FileSourceReader::get_root_folder_path() {
    return _folder_path;
}

std::vector<std::string> FileSourceReader::get_file_paths_from_meta_data_reader() {
    if (_meta_data_reader) {
        return _meta_data_reader->get_relative_file_path();
    } else {
        std::cout << "\n Meta Data Reader is not initialized!";
        return {};
    }
}

void FileSourceReader::compute_start_and_end_idx_of_all_shards() {
    for (uint shard_id = 0; shard_id < _shard_count; shard_id++) {
        auto start_idx_of_shard = (_file_count_all_shards * shard_id) / _shard_count;
        auto end_idx_of_shard = start_idx_of_shard + actual_shard_size_without_padding() - 1;
        _shard_start_idx_vector.push_back(start_idx_of_shard);
        _shard_end_idx_vector.push_back(end_idx_of_shard);
     
    }
}

size_t FileSourceReader::get_dataset_size() {
    return _file_count_all_shards;
}

size_t FileSourceReader::actual_shard_size_without_padding() {
    return std::floor((_shard_id + 1) * _file_count_all_shards / _shard_count) - std::floor(_shard_id * _file_count_all_shards / _shard_count);
}

size_t FileSourceReader::largest_shard_size_without_padding() {
    return std::ceil(_file_count_all_shards * 1.0 / _shard_count);
}