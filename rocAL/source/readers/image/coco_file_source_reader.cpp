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

#include "readers/image/coco_file_source_reader.h"
#include "meta_data/meta_data_reader_factory.h"
#include "meta_data/meta_data_graph_factory.h"
#include "pipeline/filesystem.h"

#define USE_STDIO_FILE 0

COCOFileSourceReader::COCOFileSourceReader() {
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

unsigned COCOFileSourceReader::count_items() {
    int ret;
    if (_shard_size == -1) {
        if (_loop) return shard_size_with_padding();
        int size = std::max(shard_size_with_padding(), _batch_count);
        ret = (size - _read_counter);
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

Reader::Status COCOFileSourceReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _file_id = 0;
    _folder_path = desc.path();
    _json_path = desc.json_path();
    // _file_list_path = desc.file_list_path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _last_batch_info = desc.get_last_batch_policy();
    _pad_last_batch_repeated = _last_batch_info.second;
    _stick_to_shard = true; // check and pass stick_to_shard & shard_size from user similar to audio decoders
    _shard_size = -1;
    _loop = desc.loop();
    _shuffle = desc.shuffle();
    _meta_data_reader = desc.meta_data_reader();

    if (_json_path == "") {
        std::cout << "\n _json_path has to be set manually";
        exit(0);
    }
    if (!_meta_data_reader )
        std::cout<<"Metadata reader not initialized for COCO file source\n";

    ret = subfolder_reading();
    std::cerr << "\n Initialize";
     _curr_file_idx = get_start_idx(); // shard's start_idx would vary for every shard in the vector

    if (_meta_data_reader && _meta_data_reader->get_aspect_ratio_grouping()) {
        std::cerr << "\n Initialize - ASG";
        // calculate the aspect ratio for each file and create a pair of <filename, aspect_ratio>
        std::vector<std::pair<std::string, float>> file_aspect_ratio_pair(_all_shard_file_names_padded.size());
        for (size_t i = 0; i < _all_shard_file_names_padded.size(); i++) {
            auto filename = _all_shard_file_names_padded[i];
            std::string base_filename = filename.substr(filename.find_last_of("/\\") + 1);
            auto img_size = _meta_data_reader->lookup_image_size(base_filename);
            auto aspect_ratio = static_cast<float>(img_size.h) / img_size.w;
            file_aspect_ratio_pair[i] = std::make_pair(filename, aspect_ratio);
            _aspect_ratios.push_back(aspect_ratio);
        };

        // sort the <filename, aspect_ratio> pairs according to aspect ratios
        std::sort(file_aspect_ratio_pair.begin(), file_aspect_ratio_pair.end(), [](auto &lop, auto &rop) { return lop.second < rop.second; });

        // extract sorted file_names
        std::transform(file_aspect_ratio_pair.begin(), file_aspect_ratio_pair.end(), std::back_inserter(_sorted_file_names), [](auto &pair) { return pair.first; });
        // extract sorted aspect ratios
        _aspect_ratios.clear();
        std::transform(file_aspect_ratio_pair.begin(), file_aspect_ratio_pair.end(), std::back_inserter(_aspect_ratios), [](auto &pair) { return pair.second; });

        // Copy the sorted file_names to _file_names vector to be used in sharding
        _all_shard_file_names_padded = _sorted_file_names;

        // shuffle dataset if set
        if (ret == Reader::Status::OK && _shuffle) {
            shuffle_with_aspect_ratios();
        }
        std::cerr << "\n Initialize - ASG exits";
    } else {
        // shuffle dataset if set
        if (ret == Reader::Status::OK && _shuffle)
            std::random_shuffle(_all_shard_file_names_padded.begin() + get_start_idx(), _all_shard_file_names_padded.begin() + get_start_idx() + shard_size_without_padding());
    }
    return ret;
}

void COCOFileSourceReader::increment_curr_file_idx() {
    // Should work for both pad_last_batch = True (or) False
    if (_stick_to_shard == false) {
        _curr_file_idx = (_curr_file_idx + 1) % _all_shard_file_names_padded.size();
    } else {
        if (_curr_file_idx >= get_start_idx() &&
            _curr_file_idx < get_start_idx() + shard_size_without_padding() - 1) // checking if current-element lies within the shard size [begin_idx, last_idx -1]
            _curr_file_idx = (_curr_file_idx + 1);
        else
            _curr_file_idx = get_start_idx();
    }
}

void COCOFileSourceReader::incremenet_read_ptr() {
    _read_counter++;
    increment_curr_file_idx();
}

size_t COCOFileSourceReader::open() {
    auto file_path = _all_shard_file_names_padded[_curr_file_idx];  // Get next file name
    incremenet_read_ptr();
    _last_id = file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        _last_id.erase(0, last_slash_idx + 1);
    }

#if USE_STDIO_FILE
    _current_fPtr = fopen(file_path.c_str(), "rb");  // Open the file,
    if (!_current_fPtr)                              // Check if it is ready for reading
        return 0;
    fseek(_current_fPtr, 0, SEEK_END);          // Take the file read pointer to the end
    _current_file_size = ftell(_current_fPtr);  // Check how many bytes are there between and the current read pointer position (end of the file)
    if (_current_file_size == 0) {              // If file is empty continue
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        return 0;
    }
    fseek(_current_fPtr, 0, SEEK_SET);  // Take the file pointer back to the start
#else
    _current_ifs.open(file_path, std::ifstream::in);
    if (_current_ifs.fail()) return 0;
    // Determine the file length
    _current_ifs.seekg(0, std::ios_base::end);
    _current_file_size = _current_ifs.tellg();
    if (_current_file_size == 0) {  // If file is empty continue
        _current_ifs.close();
        return 0;
    }
    _current_ifs.seekg(0, std::ios_base::beg);
#endif
    return _current_file_size;
}

size_t COCOFileSourceReader::read_data(unsigned char *buf, size_t read_size) {
#if USE_STDIO_FILE
    if (!_current_fPtr)
        return 0;
#else
    if (!_current_ifs)
        return 0;
#endif
    // Requested read size bigger than the file size? just read as many bytes as the file size
    read_size = (read_size > _current_file_size) ? _current_file_size : read_size;
#if USE_STDIO_FILE
    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
#else
    _current_ifs.read((char *)buf, (int)read_size);
    size_t actual_read_size = _current_ifs.gcount();
#endif
    return actual_read_size;
}

int COCOFileSourceReader::close() {
    return release();
}

COCOFileSourceReader::~COCOFileSourceReader() {
    release();
}

int COCOFileSourceReader::release() {
#if USE_STDIO_FILE
    if (!_current_fPtr)
        return 0;
    fclose(_current_fPtr);
    _current_fPtr = nullptr;
#else
    if (_current_ifs.bad())
        return 0;
    _current_ifs.close();
#endif
    return 0;
}

void COCOFileSourceReader::shuffle_with_aspect_ratios() {
    std::cerr << "\n Shuffling with aspect ratios";
    // Calculate the mid element which divides the aspect ratios into two groups (<=1.0 and >1.0)
    auto mid = std::upper_bound(_aspect_ratios.begin() + get_start_idx(), _aspect_ratios.begin() + get_start_idx() + shard_size_without_padding(), 1.0f) - (_aspect_ratios.begin() + get_start_idx());
    // // Shuffle within groups using the mid element as the limit - [start, mid) and [mid, last)
    std::cerr << "\n mid" << mid;
    std::random_shuffle(_all_shard_file_names_padded.begin() + get_start_idx(), _all_shard_file_names_padded.begin() + get_start_idx() + mid);
    // std::random_shuffle(_all_shard_file_names_padded.begin() + get_start_idx() + mid, _all_shard_file_names_padded.begin() + get_start_idx() + shard_size_without_padding());
    // std::vector<std::string> shuffled_filenames;
    // int split_count = (_all_shard_file_names_padded.size() /_shard_count) / _batch_count;  // Number of batches for this shard
    // std::vector<int> indexes(split_count);
    // std::iota(indexes.begin(), indexes.end(), 0);
    // // Shuffle the index vector and use the index to fetch batch size elements for decoding
    // std::random_shuffle(indexes.begin(), indexes.end());
    // std::cerr << "\n Shuffling with aspect ratios - indexs logic 1";
    // for (auto const idx : indexes)
    //     shuffled_filenames.insert(shuffled_filenames.end(), _all_shard_file_names_padded.begin() + get_start_idx() + idx * _batch_count, _all_shard_file_names_padded.begin() + get_start_idx() + idx * _batch_count + _batch_count);
    // // uint i = 0;
    // // std::cerr << "\n Shuffling with aspect ratios - indexs logic 2";
    // // for (auto &ele: shuffled_filenames) {
    // //     _all_shard_file_names_padded.insert(_all_shard_file_names_padded.begin() + get_start_idx() + i, ele);
    // //     i++;
    // // }
    // std::copy(_all_shard_file_names_padded.begin() + get_start_idx(), _all_shard_file_names_padded.begin() + get_start_idx() + shard_size_without_padding(), std::back_inserter(shuffled_filenames));
    std::cerr << "\n Shuffling with aspect ratios - indexs logic 3";
    // _all_shard_file_names_padded = shuffled_filenames;
    std::cerr << "\n Shuffling with aspect ratios is done";
}

void COCOFileSourceReader::reset() {

    if (_stick_to_shard == false)
        increment_shard_id(); // Should work for both single and multiple shards

    if (_meta_data_reader && _meta_data_reader->get_aspect_ratio_grouping()) {
        _all_shard_file_names_padded = _sorted_file_names;
        if (_shuffle) shuffle_with_aspect_ratios();
    } else if (_shuffle) {
        std::random_shuffle(_all_shard_file_names_padded.begin() + get_start_idx(),
                            _all_shard_file_names_padded.begin() + get_start_idx() + shard_size_without_padding());
    }

    _read_counter = 0;

    if (_last_batch_info.first == RocalBatchPolicy::DROP) { // Skipping the dropped batch in next epoch
        for (uint i = 0; i < _batch_count; i++)
            increment_curr_file_idx();
    }
}

Reader::Status COCOFileSourceReader::subfolder_reading() {
    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;

    while ((_entity = readdir(_sub_dir)) != nullptr) {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
            continue;
        entry_name_list.push_back(entry_name);
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
                WRN("FileReader ShardID [" + TOSTR(_shard_id) + "] File reader cannot access the storage at " + _folder_path);
        }
    }
    if (!_file_names.empty())
        LOG("FileReader ShardID [" + TOSTR(_shard_id) + "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path)
    closedir(_sub_dir);

    auto dataset_size = _file_count_all_shards;
    // Pad the _file_names with last element of the shard in the vector when _pad_last_batch_repeated is True
    if (_shard_size > 0)
        _padded_samples = _shard_size % _batch_count;
    else
        _padded_samples = shard_size_with_padding() % _batch_count;
    _last_batch_padded_size = _batch_count - _padded_samples;

    if (_pad_last_batch_repeated ==
        true) { // pad the last sample when the dataset_size is not divisible by
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

    std::cerr << "\n Sub folder reading is done";
    return ret;
}

Reader::Status COCOFileSourceReader::open_folder() {
    if ((_src_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        if (_entity->d_type != DT_REG)
            continue;
        if (!_meta_data_reader || _meta_data_reader->exists(_entity->d_name)) {
            std::string file_path = _folder_path;
            file_path.append("/");
            file_path.append(_entity->d_name);
            _file_names.push_back(file_path);
            _last_file_name = file_path;
            _file_count_all_shards++;
            incremenet_file_id();
        } else {
            WRN("Skipping file," + _entity->d_name + " as it is not present in metadata reader")
        }
    }

    if (_file_names.empty())
        WRN("FileReader ShardID [" + TOSTR(_shard_id) + "] Did not load any file from " + _folder_path)
    std::sort(_file_names.begin(), _file_names.end());

    closedir(_src_dir);
    return Reader::Status::OK;
}

size_t COCOFileSourceReader::get_file_shard_id() {
    if (_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    // return (_file_id / (_batch_count)) % _shard_count;
    return _file_id % _shard_count;
}

std::string COCOFileSourceReader::get_root_folder_path() {
    return _folder_path;
}

std::vector<std::string> COCOFileSourceReader::get_file_paths_from_meta_data_reader() {   if (_meta_data_reader) {
        return _meta_data_reader->get_file_path_content();
    } else {
        std::clog << "\n Meta Data Reader is not initialized!";
        return {};
    }
}

size_t COCOFileSourceReader::last_batch_padded_size() {
    return _last_batch_padded_size;
}

size_t COCOFileSourceReader::get_start_idx() {
    _shard_start_idx = (get_dataset_size() * _shard_id) / _shard_count;
    return _shard_start_idx;
}

size_t COCOFileSourceReader::get_dataset_size() {
    return _file_count_all_shards;
}


size_t COCOFileSourceReader::shard_size_without_padding() {
    return std::floor((_shard_id + 1) * get_dataset_size() / _shard_count) - floor(_shard_id * get_dataset_size() / _shard_count);
}

size_t COCOFileSourceReader::shard_size_with_padding() {
  return std::ceil(get_dataset_size() * 1.0 / _shard_count);
}

void COCOFileSourceReader::increment_shard_id() {
    _shard_id = (_shard_id + 1) % _shard_count;
}
