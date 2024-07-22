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

#include "readers/image/mxnet_recordio_reader.h"

#include "pipeline/commons.h"
#include <memory.h>
#include <math.h>
#include <stdint.h>
#include "readers/image/mxnet_recordio_reader.h"
#include "pipeline/filesystem.h"

using namespace std;

MXNetRecordIOReader::MXNetRecordIOReader() {
    _src_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _loop = false;
    _shuffle = false;
    _file_count_all_shards = 0;
}

unsigned MXNetRecordIOReader::count_items() {
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

Reader::Status MXNetRecordIOReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _loop = desc.loop();
    _shuffle = desc.shuffle();
    _pad_last_batch_repeated = _last_batch_info.second;
    _stick_to_shard = desc.get_stick_to_shard();
    _shard_size = desc.get_shard_size();
    ret = record_reading();
    _curr_file_idx = get_start_idx(); // shard's start_idx would vary for every shard in the vector

    // shuffle dataset if set
    if (ret == Reader::Status::OK && _shuffle)
        std::random_shuffle(_all_shard_file_names_padded.begin() + get_start_idx(),
                            _all_shard_file_names_padded.begin() + get_start_idx() + actual_shard_size_without_padding());

    return ret;
}

void MXNetRecordIOReader::increment_curr_file_idx() {
    // Should work for both pad_last_batch = True (or) False
    if (_stick_to_shard == false) {
        _curr_file_idx = (_curr_file_idx + 1) % _all_shard_file_names_padded.size();
    } else {
        if (_curr_file_idx >= get_start_idx() &&
            _curr_file_idx < get_start_idx() + actual_shard_size_without_padding() - 1) // checking if current-element lies within the shard size [begin_idx, last_idx -1]
            _curr_file_idx = (_curr_file_idx + 1);
        else
            _curr_file_idx = get_start_idx();
    }
}

void MXNetRecordIOReader::incremenet_read_ptr() {
    _read_counter++;
    increment_curr_file_idx();
}

size_t MXNetRecordIOReader::open() {
    auto file_path = _file_names[_curr_file_idx];  // Get next file name
    _last_id = file_path;
    auto it = _record_properties.find(_file_names[_curr_file_idx]);
    std::tie(_current_file_size, _seek_pos, _data_size_to_read) = it->second;
    return _current_file_size;
}

size_t MXNetRecordIOReader::read_data(unsigned char *buf, size_t read_size) {
    auto it = _record_properties.find(_file_names[_curr_file_idx]);
    std::tie(_current_file_size, _seek_pos, _data_size_to_read) = it->second;
    read_image(buf, _seek_pos, _data_size_to_read);
    incremenet_read_ptr();
    return read_size;
}

int MXNetRecordIOReader::close() {
    return release();
}

MXNetRecordIOReader::~MXNetRecordIOReader() {
}

int MXNetRecordIOReader::release() {
    return 0;
}

void MXNetRecordIOReader::reset() {
    if (_shuffle)
        std::random_shuffle(_all_shard_file_names_padded.begin() + get_start_idx(),
                            _all_shard_file_names_padded.begin() + get_start_idx() + actual_shard_size_without_padding());

    if (_stick_to_shard == false) // Pick elements from the next shard - hence increment shard_id
        increment_shard_id();     // Should work for both single and multiple shards

    _read_counter = 0;

    if (_last_batch_info.first == RocalBatchPolicy::DROP) { // Skipping the dropped batch in next epoch
        for (uint i = 0; i < _batch_count; i++)
            increment_curr_file_idx();
    }
}

Reader::Status MXNetRecordIOReader::record_reading() {
    auto ret = Reader::Status::OK;
    if (MXNet_reader() != Reader::Status::OK)
        WRN("MXNetRecordIOReader ShardID [" + TOSTR(_shard_id) + "] MXNetRecordIOReader cannot access the storage at " + _path);

    if (!_file_names.empty())
        LOG("MXNetRecordIOReader ShardID [" << TOSTR(_shard_id) << "] Total of " << TOSTR(_file_names.size()) << " images loaded from " << _path)
    
        auto dataset_size = _file_count_all_shards;
    // Pad the _file_names with last element of the shard in the vector when _pad_last_batch_repeated is True
    if (_shard_size > 0)
        _padded_samples = _shard_size % _batch_count;
    else
        _padded_samples = largest_shard_size_without_padding() % _batch_count;
    if (_padded_samples != 0)
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
            if (start != end && start <= _file_names.end() &&
                end <= _file_names.end()) {
                _all_shard_file_names_padded.insert(_all_shard_file_names_padded.end(), start, end);
            }
            if (shard_size_with_padding % _batch_count) {
                _num_padded_samples = (shard_size_with_padding - shard_size_without_padding) + _batch_count - (shard_size_with_padding % _batch_count);
                _file_count_all_shards += _num_padded_samples;
                _all_shard_file_names_padded.insert(_all_shard_file_names_padded.end(), _num_padded_samples, _all_shard_file_names_padded.back());
            }
        }
    } else {
        _all_shard_file_names_padded = _file_names;
    }
    _last_file_name = _all_shard_file_names_padded[_all_shard_file_names_padded.size() - 1];
    return ret;
}

Reader::Status MXNetRecordIOReader::MXNet_reader() {
    std::string _rec_file, _idx_file;
    if ((_src_dir = opendir(_path.c_str())) == nullptr)
        THROW("MXNetReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        std::string file_name = _path + "/" + _entity->d_name;
        filesys::path pathObj(file_name);
        if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
            auto file_extension_idx = file_name.find_last_of(".");
            if (file_extension_idx != std::string::npos) {
                std::string file_extension = file_name.substr(file_extension_idx + 1);
                if (file_extension == "rec")
                    _rec_file = file_name;
                else if (file_extension == "idx")
                    _idx_file = file_name;
                else
                    continue;
            }
        }
    }
    closedir(_src_dir);
    uint rec_size;
    _file_contents.open(_rec_file, ios::binary);
    if (!_file_contents)
        THROW("MXNetRecordIOReader ERROR: Failed opening the file " + _rec_file);
    _file_contents.seekg(0, ifstream::end);
    rec_size = _file_contents.tellg();
    _file_contents.seekg(0, ifstream::beg);

    ifstream index_file(_idx_file);
    if (!index_file)
        THROW("MXNetRecordIOReader ERROR: Could not open RecordIO index file. Provided path: " + _idx_file);

    std::vector<size_t> _index_list;
    size_t _index, _offset;
    while (index_file >> _index >> _offset)
        _index_list.push_back(_offset);
    if (_index_list.empty())
        THROW("MXNetRecordIOReader ERROR: RecordIO index file doesn't contain any indices. Provided path: " + _idx_file);
    _index_list.push_back(rec_size);
    std::sort(_index_list.begin(), _index_list.end());
    for (size_t i = 0; i < _index_list.size() - 1; ++i)
        _indices.emplace_back(_index_list[i], _index_list[i + 1] - _index_list[i]);
    read_image_names();
    return Reader::Status::OK;
}

void MXNetRecordIOReader::read_image_names() {
    for (int current_index = 0; current_index < (int)_indices.size(); current_index++) {
        uint32_t _magic, _length_flag;
        std::tie(_seek_pos, _data_size_to_read) = _indices[current_index];
        _file_contents.seekg(_seek_pos, ifstream::beg);
        uint8_t *_data = new uint8_t[_data_size_to_read];
        uint8_t *_data_ptr = _data;
        auto ret = _file_contents.read((char *)_data_ptr, _data_size_to_read).gcount();
        if (ret == -1 || ret != _data_size_to_read)
            THROW("MXNetRecordIOReader ERROR:  Unable to read the data from the file ");

        _magic = *((uint32_t *)_data_ptr);
        _data_ptr += sizeof(_magic);
        if (_magic != _kMagic)
            THROW("MXNetRecordIOReader ERROR: Invalid MXNet RecordIO: wrong _magic number");
        _length_flag = *((uint32_t *)_data_ptr);
        _data_ptr += sizeof(_length_flag);
        uint32_t _clength = DecodeLength(_length_flag);
        _hdr = *((ImageRecordIOHeader *)_data_ptr);

        if (_hdr.flag == 0)
            _image_key = to_string(_hdr.image_id[0]);
        else {
            WRN("\nMXNetRecordIOReader Multiple record reading has not supported");
            continue;
        }
        /* _clength - sizeof(ImageRecordIOHeader) to get the data size.
        Subtracting label size(_hdr.flag * sizeof(float)) from data size to get image size*/
        int64_t image_size = (_clength - sizeof(ImageRecordIOHeader)) - (_hdr.flag * sizeof(float));
        delete[] _data;

        _file_names.push_back(_image_key.c_str());
        _last_file_name = _image_key.c_str();

        _file_count_all_shards++;

        _last_file_size = image_size;
        _last_seek_pos = _seek_pos;
        _last_data_size = _data_size_to_read;
        //_record_properties vector used to keep track of image size, seek position and data size of the single
        _record_properties.insert(pair<std::string, std::tuple<unsigned int, int64_t, int64_t>>(_last_file_name, std::make_tuple(_last_file_size, _last_seek_pos, _last_data_size)));
    }
}

void MXNetRecordIOReader::read_image(unsigned char *buff, int64_t seek_position, int64_t _data_size_to_read) {
    uint32_t _magic, _length_flag;
    _file_contents.seekg(seek_position, ifstream::beg);
    uint8_t *_data = new uint8_t[_data_size_to_read];
    uint8_t *_data_ptr = _data;
    auto ret = _file_contents.read((char *)_data_ptr, _data_size_to_read).gcount();
    if (ret == -1 || ret != _data_size_to_read)
        THROW("MXNetRecordIOReader ERROR:  Unable to read the data from the file ");
    _magic = *((uint32_t *)_data_ptr);
    _data_ptr += sizeof(_magic);
    if (_magic != _kMagic)
        THROW("MXNetRecordIOReader ERROR: Invalid RecordIO: wrong _magic number");
    _length_flag = *((uint32_t *)_data_ptr);
    _data_ptr += sizeof(_length_flag);
    uint32_t _cflag = DecodeFlag(_length_flag);
    uint32_t _clength = DecodeLength(_length_flag);
    _hdr = *((ImageRecordIOHeader *)_data_ptr);
    _data_ptr += sizeof(_hdr);

    int64_t data_size = _clength - sizeof(ImageRecordIOHeader);
    int64_t label_size = _hdr.flag * sizeof(float);
    int64_t image_size = data_size - label_size;
    if (_cflag == 0)
        memcpy(buff, _data_ptr + label_size, image_size);
    else
        THROW("\nMultiple record reading has not supported");
    delete[] _data;
}

size_t MXNetRecordIOReader::last_batch_padded_size() {
    return _last_batch_padded_size;
}

size_t MXNetRecordIOReader::get_start_idx() {
    _shard_start_idx = (get_dataset_size() * _shard_id) / _shard_count;
    return _shard_start_idx;
}

size_t MXNetRecordIOReader::get_dataset_size() {
    return _file_count_all_shards;
}


size_t MXNetRecordIOReader::actual_shard_size_without_padding() {
    return std::floor((_shard_id + 1) * get_dataset_size() / _shard_count) - floor(_shard_id * get_dataset_size() / _shard_count);
}

size_t MXNetRecordIOReader::largest_shard_size_without_padding() {
  return std::ceil(get_dataset_size() * 1.0 / _shard_count);
}

void MXNetRecordIOReader::increment_shard_id() {
    _shard_id = (_shard_id + 1) % _shard_count;
}
