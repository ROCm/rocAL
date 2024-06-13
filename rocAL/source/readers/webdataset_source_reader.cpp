uu
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
    _record_name_prefix = "";
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
    _path = desc.path(); // TODO: need to add support fo the index file yet
    _feature_key_map = desc.feature_key_map();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _loop = desc.loop();
    _shuffle = desc.shuffle();
    _record_name_prefix = desc.file_prefix();
    _encoded_key = _feature_key_map.at("image/encoded");
    _filename_key = _feature_key_map.at("image/filename");
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
    auto ret = read_image(buf, _file_names[_curr_file_idx], _file_size[_file_names[_curr_file_idx]], _file_offset_file_names[_curr_file_idx]);
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

Reader::Status WebDatasetSourceReader::folder_reading() {
    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

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
    for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
        std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
        _folder_path = subfolder_path;
        if (webdataset_record_reader() != Reader::Status::OK)
            WRN("FileReader ShardID [" + TOSTR(_shard_id) + "] File reader cannot access the storage at " + _folder_path);
    }

    if (!_file_names.empty())
        LOG("FileReader ShardID [" + TOSTR(_shard_id) + "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path)
    closedir(_sub_dir);
    return ret;
}


Reader::Status WebDatasetSourceReader::webdataset_record_reader() {
    // Open the webdataset file
    std::string tar_file_path = _folder_path;
    TAR* tar;
    tar_open(&tar, tar_filename, NULL, O_RDONLY, 0, 0);

    int result;
    off_t offset = 0; // Initialize offset
    while ((result = th_read(tar)) == 0) {
        // Get file information
        char* filename = th_get_pathname(tar);
        size_t size = th_get_size(tar);

        // Allocate buffer for file content
        char* buffer = new char[size + 1];
        buffer[size] = '\0';

        // Read file content
        size_t total_read = 0;
        while (total_read < size) {
            size_t bytes_to_read = std::min<size_t>(BLOCKSIZE, size - total_read);
            ssize_t bytes_read = tar_read(tar, buffer + total_read, bytes_to_read);
            if (bytes_read < 0) {
                std::cerr << "Error reading file from tar - Webdataset Reader: " << strerror(errno) << std::endl;
                delete[] buffer;
                tar_close(tar);
                return Reader::Status::ERROR; // Return error status
            }
            total_read += bytes_read;
        }
        
        // Calculate offset
        off_t file_offset = th_get_offset(tar);

        // Update file path and size
        std::string file_path = _folder_path;
        file_path.append("/");
        file_path.append(filename);
        _file_names.push_back(file_path);
        _file_size.insert(std::pair<std::string, unsigned int>(file_path, size));
        _file_tar_mapping.insert(std::pair<std::string, std::string>(file_path, _folder_path));
        _file_offset.insert(std::pair<std::string, unsigned int>(file_path, file_offset));

        // Update overall offset
        offset += th_get_size(tar) + th_get_offset(tar);
    }

    // Close the tar file
    tar_close(tar);

    return Reader::Status::OK; // Return OK status
}


size_t WebDatasetSourceReader::get_file_shard_id() {
    if (_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    return _file_id % _shard_count;
}

// Reader::Status WebDatasetSourceReader::read_image_names(std::ifstream &file_contents, uint file_size) {
//     auto ret = Reader::Status::OK;
//     while (!_last_rec) {

//     return ret;
// }

Reader::Status WebDatasetSourceReader::read_image(unsigned char *buff, std::string file_name, uint file_size, uint offset) {
    auto ret = Reader::Status::OK;
    TAR* tar;
    // Extract tar name from the file name
    auto tar_file_name  = _file_tar_mapping[file_name];
    tar_open(&tar, tar_file_name, NULL, O_RDONLY, 0, 0);
    if (!tar) {
        std::cerr << "Failed to open tar file: " << tar_file_name << std::endl;
        return false;
    }

    // Seek to the specified offset within the tar file
    if (tar_seek(tar, offset, SEEK_SET) == -1) {
        std::cerr << "Failed to seek to offset: " << offset << std::endl;
        tar_close(tar);
        return false;
    }

    // Read the specified number of bytes
    size_t total_read = 0;
    while (total_read < size) {
        size_t bytes_to_read = std::min<size_t>(BLOCKSIZE, size - total_read);
        ssize_t bytes_read = tar_read(tar, buff + total_read, bytes_to_read);
        if (bytes_read < 0) {
            std::cerr << "Error reading file from tar: " << strerror(errno) << std::endl;
            tar_close(tar);
            return false;
        }
        total_read += bytes_read;
    }

    // Close the tar file
    tar_close(tar);
    return ret;
}
