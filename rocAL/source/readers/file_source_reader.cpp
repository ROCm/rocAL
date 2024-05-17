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
    if (_loop)
        return _file_names.size();

    int ret = ((int)_file_names.size() - _read_counter);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status FileSourceReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _file_id = 0;
    _folder_path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    _meta_data_reader = desc.meta_data_reader();
    _last_batch_info = desc.get_last_batch_policy();
    ret = subfolder_reading();
    // shuffle dataset if set
    if (ret == Reader::Status::OK && _shuffle)
        std::random_shuffle(_file_names.begin(), _file_names.end());

    return ret;
}

void FileSourceReader::incremenet_read_ptr() {
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
    if (_last_batch_info.first == RocalBatchPolicy::DROP) {
        if (_last_batch_info.second == true) {
            // Check for the last batch and skip it by incrementing with batch_size - hence dropping the last batch
            if ((_file_names.size() / _batch_count) == _curr_file_idx)  // To check if it the last batch
            {
                _curr_file_idx += _batch_count; // This increaments the ptr with batch size - meaning the batch is skipped.
                _curr_file_idx = (_curr_file_idx + 1) % _file_names.size(); // When the last_batch_pad is true, next iter should start from beginning. This line ensures, pointer from end is brought back to beginning.
            }
        } else {
            THROW("Not implemented");
        }
    }
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
    if (_shuffle) std::random_shuffle(_file_names.begin(), _file_names.end());
    _read_counter = 0;
    _curr_file_idx = 0;
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

    if (_file_names.empty())
        ERR("FileReader ShardID [" + TOSTR(_shard_id) + "] Did not load any file from " + _folder_path)

    // the following code is required to make every shard the same size - required for the multi-gpu training
    uint images_to_pad_shard = (ceil(_file_count_all_shards / _shard_count) * _shard_count) - _file_count_all_shards;
    if (!images_to_pad_shard) {
        for (uint i = 0; i < images_to_pad_shard; i++) {
            if (get_file_shard_id() != _shard_id) {
                _file_count_all_shards++;
                incremenet_file_id();
                continue;
            }
            _last_file_name = _file_names.at(i);
            _file_names.push_back(_last_file_name);
            _file_count_all_shards++;
            incremenet_file_id();
        }
    }

    return ret;
}

Reader::Status FileSourceReader::subfolder_reading() {
    auto ret = generate_file_names();

    if (_in_batch_read_count > 0 && _in_batch_read_count < _batch_count) {
        // This is to pad within a batch in a shard. Need to change this according to fill / drop or partial.
        // Adjust last batch only if the last batch padded is true.
        fill_last_batch();
        LOG("FileReader ShardID [" + TOSTR(_shard_id) + "] Replicated " + _folder_path + _last_file_name + " " + TOSTR((_batch_count - _in_batch_read_count)) + " times to fill the last batch")
    }
    if (!_file_names.empty())
        LOG("FileReader ShardID [" + TOSTR(_shard_id) + "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path)
    return ret;
}

void FileSourceReader::fill_last_batch() {
    if (_last_batch_info.first == RocalBatchPolicy::FILL || _last_batch_info.first == RocalBatchPolicy::PARTIAL) {
        if (_last_batch_info.second == true) {
            for (size_t i = 0; i < (_batch_count - _in_batch_read_count); i++)
                _file_names.push_back(_last_file_name);
        } else {
            THROW("Not implemented");
        }
    } else if (_last_batch_info.first == RocalBatchPolicy::DROP) {
        for (size_t i = 0; i < _in_batch_read_count; i++)
            _file_names.pop_back();
    }
    if (_last_batch_info.first == RocalBatchPolicy::PARTIAL) {
        _last_batch_padded_size = _batch_count - _in_batch_read_count;
    }
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
            if (get_file_shard_id() != _shard_id) {
                _file_count_all_shards++;
                incremenet_file_id();
                continue;
            }
            _in_batch_read_count++;
            _in_batch_read_count = (_in_batch_read_count % _batch_count == 0) ? 0 : _in_batch_read_count;
            _file_names.push_back(file_path);
            _file_count_all_shards++;
            incremenet_file_id();
        } else {
            WRN("Skipping file," + filename + " as it is not present in metadata reader")
        }
    }
    if (_file_names.empty())
        ERR("FileReader ShardID [" + TOSTR(_shard_id) + "] Did not load any file from " + _folder_path)
    _last_file_name = _file_names[_file_names.size() - 1];

    closedir(_src_dir);
    return Reader::Status::OK;
}

size_t FileSourceReader::get_file_shard_id() {
    if (_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    return _file_id % _shard_count;
}

size_t FileSourceReader::last_batch_padded_size() {
    return _last_batch_padded_size;
}
