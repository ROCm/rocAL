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

#include <math.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <random>

#include "pipeline/commons.h"
#include "pipeline/filesystem.h"
#include "readers/image/numpy_data_reader.h"

NumpyDataReader::NumpyDataReader() {
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
    _shuffle = false;
    _file_count_all_shards = 0;
}

unsigned NumpyDataReader::count_items() {
    int size = get_max_size_of_shard(_batch_size, _loop);
    int ret = (size - _read_counter);
    if (_sharding_info.last_batch_policy == RocalBatchPolicy::DROP && _last_batch_padded_size != 0)
        ret -= _batch_size;
    return ((ret < 0) ? 0 : ret);
}

Reader::Status NumpyDataReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _folder_path = desc.path();
    _file_list_path = desc.file_list_path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_size = desc.get_batch_size();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    _meta_data_reader = desc.meta_data_reader();
    _sharding_info = desc.get_sharding_info();
    _pad_last_batch_repeated = _sharding_info.pad_last_batch_repeated;
    _stick_to_shard = _sharding_info.stick_to_shard;
    _shard_size = _sharding_info.shard_size;
    _files = desc.get_files();
    _seed = desc.seed();
    ret = subfolder_reading();
    _file_headers.resize(_file_names.size());
    // shuffle dataset if set
    if (ret == Reader::Status::OK && _shuffle) {
        std::mt19937 rng(_seed);
        std::shuffle(_file_names.begin() + _shard_start_idx_vector[_shard_id],
                     _file_names.begin() + _shard_end_idx_vector[_shard_id], rng);
    }
    return ret;
}

void NumpyDataReader::incremenet_read_ptr() {
    _read_counter++;
    increment_curr_file_idx(_file_names.size());
}

size_t NumpyDataReader::open() {
    auto file_path = _file_names[_curr_file_idx];       // Get current file name
    _curr_file_header = _file_headers[_curr_file_idx];  // Get current file header
    incremenet_read_ptr();
    _last_file_path = _last_id = file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        _last_id.erase(0, last_slash_idx + 1);
    }

    auto ret = get_header_from_cache(file_path, _curr_file_header);
    if (!ret) {
        parse_header(_curr_file_header, file_path);
        update_header_cache(file_path, _curr_file_header);
    } else {
        _current_fPtr = std::fopen(file_path.c_str(), "rb");
        if (_current_fPtr == nullptr)
            THROW("Could not open file " + file_path + ": " + std::strerror(errno));
    }
    fseek(_current_fPtr, 0, SEEK_SET);  // Take the file pointer back to the start

    return _curr_file_header.nbytes();
}

bool NumpyDataReader::get_header_from_cache(const std::string& file_name, NumpyHeaderData& header) {
    std::unique_lock<std::mutex> cache_lock(_cache_mutex);
    auto it = _header_cache.find(file_name);
    if (it == _header_cache.end()) {
        return false;
    } else {
        header = it->second;
        return true;
    }
}

void NumpyDataReader::update_header_cache(const std::string& file_name, const NumpyHeaderData& value) {
    std::unique_lock<std::mutex> cache_lock(_cache_mutex);
    _header_cache[file_name] = value;
}

const RocalTensorDataType NumpyDataReader::get_dtype(const std::string& format) {
    if (format == "u1") return RocalTensorDataType::UINT8;
    if (format == "u2") THROW("uint16_t dtype not supported in rocAL");
    if (format == "u4") return RocalTensorDataType::UINT32;
    if (format == "u8") THROW("uint64_t dtype not supported in rocAL");
    if (format == "i1") return RocalTensorDataType::INT8;
    if (format == "i2") return RocalTensorDataType::INT16;
    if (format == "i4") return RocalTensorDataType::INT32;
    if (format == "i8") THROW("int64_t dtype not supported in rocAL");
    if (format == "f2")
#if defined(AMD_FP16_SUPPORT)
        return RocalTensorDataType::FP16;
#else
        THROW("FLOAT16 type tensor not supported")
#endif
    if (format == "f4") return RocalTensorDataType::FP32;
    if (format == "f8") THROW("double dtype not supported in rocAL");
    THROW("Unknown Numpy dtype string");
}

inline void NumpyDataReader::skip_spaces(const char*& ptr) {
    while (::isspace(*ptr))
        ptr++;
}

template <size_t N>
void NumpyDataReader::skip_char(const char*& ptr, const char (&what)[N]) {
    if (strncmp(ptr, what, N - 1))
        THROW("Found wrong symbol during parsing");
    ptr += N - 1;
}

template <size_t N>
bool NumpyDataReader::try_skip_char(const char*& ptr, const char (&what)[N]) {
    if (!strncmp(ptr, what, N - 1)) {
        ptr += N - 1;
        return true;
    } else {
        return false;
    }
}

template <size_t N>
void NumpyDataReader::skip_field(const char*& ptr, const char (&name)[N]) {
    skip_spaces(ptr);
    skip_char(ptr, "'");
    skip_char(ptr, name);
    skip_char(ptr, "'");
    skip_spaces(ptr);
    skip_char(ptr, ":");
    skip_spaces(ptr);
}

template <typename T = int64_t>
T NumpyDataReader::parse_int(const char*& ptr) {
    char* out_ptr = const_cast<char*>(ptr);  // strtol takes a non-const pointer
    T value = static_cast<T>(strtol(ptr, &out_ptr, 10));
    if (out_ptr == ptr)
        THROW("Parse error: expected a number.");
    ptr = out_ptr;
    return value;
}

std::string NumpyDataReader::parse_string(const char*& input, char delim_start, char delim_end) {
    if (*input++ != delim_start)
        THROW("Expected \'" + std::to_string(delim_start) + "\'");
    std::string out;
    for (; *input != '\0'; input++) {
        if (*input == '\\') {
            switch (*++input) {
                case '\\':
                    out += '\\';
                    break;
                case '\'':
                    out += '\'';
                    break;
                case '\t':
                    out += '\t';
                    break;
                case '\n':
                    out += '\n';
                    break;
                case '\"':
                    out += '\"';
                    break;
                default:
                    out += '\\';
                    out += *input;
                    break;
            }
        } else if (*input == delim_end) {
            break;
        } else {
            out += *input;
        }
    }
    if (*input++ != delim_end)
        THROW("Expected \'" + std::to_string(delim_end) + "\'");
    return out;
}

void NumpyDataReader::parse_header_data(NumpyHeaderData& target, const std::string& header) {
    const char* hdr = header.c_str();
    skip_spaces(hdr);
    skip_char(hdr, "{");
    skip_field(hdr, "descr");
    auto typestr = parse_string(hdr);
    // < means LE, | means N/A, = means native. In all those cases, we can read
    bool little_endian = (typestr[0] == '<' || typestr[0] == '|' || typestr[0] == '=');
    if (!little_endian)
        THROW("Big Endian files are not supported.");
    target.type_info = get_dtype(typestr.substr(1));

    skip_spaces(hdr);
    skip_char(hdr, ",");
    skip_field(hdr, "fortran_order");
    if (try_skip_char(hdr, "True")) {
        target.fortran_order = true;
    } else if (try_skip_char(hdr, "False")) {
        target.fortran_order = false;
    } else {
        THROW("Failed to parse fortran_order field.");
    }
    skip_spaces(hdr);
    skip_char(hdr, ",");
    skip_field(hdr, "shape");
    skip_char(hdr, "(");
    skip_spaces(hdr);
    target.array_shape.clear();
    while (*hdr != ')') {
        // parse_int already skips the leading spaces (strtol does).
        target.array_shape.push_back(static_cast<unsigned>(parse_int<int64_t>(hdr)));
        skip_spaces(hdr);
        if (!(try_skip_char(hdr, ",")) && (target.array_shape.size() <= 1))
            THROW("The first number in a tuple must be followed by a comma.");
    }
    if (target.fortran_order) {
        // cheapest thing to do is to define the tensor in an reversed way
        std::reverse(target.array_shape.begin(), target.array_shape.end());
    }
}

void NumpyDataReader::parse_header(NumpyHeaderData& parsed_header, std::string file_path) {
    // check if the file is actually a numpy file
    std::vector<char> token(128);
    _current_fPtr = std::fopen(file_path.c_str(), "rb");
    if (_current_fPtr == nullptr)
        THROW("Could not open file " + file_path + ": " + std::strerror(errno));
    int64_t n_read = std::fread(token.data(), 1, 10, _current_fPtr);
    if (n_read != 10)
        THROW("Can not read header.");
    token[n_read] = '\0';

    // rocAL does not support numpy V2 headers
    // https://numpy.org/neps/nep-0001-npy-format.html
    int np_api_version = token[6];
    if (np_api_version != 1)
        THROW("rocAL only supports reading npy files with NPY file format version 1");

    // check if header is too short
    std::string header = std::string(token.data());
    if (header.find_first_of("NUMPY") == std::string::npos)
        THROW("File is not a numpy file.");

    // extract header length which can have up to 65535 bytes
    uint16_t header_len = 0;
    memcpy(&header_len, &token[8], 2);
    if ((header_len + 10) % 16 != 0)
        THROW("Error extracting header length.");

    // read header: the offset is a magic number
    int64_t offset = 6 + 1 + 1 + 2;
    token.resize(header_len + 1);
    if (std::fseek(_current_fPtr, offset, SEEK_SET))
        THROW("Seek operation failed: " + std::strerror(errno));
    n_read = std::fread(token.data(), 1, header_len, _current_fPtr);
    if (n_read != header_len)
        THROW("Can not read header.");
    token[header_len] = '\0';
    header = std::string(token.data());
    if (header.find('{') == std::string::npos)
        THROW("Header is corrupted.");
    offset += header_len;
    if (std::fseek(_current_fPtr, offset, SEEK_SET))
        THROW("Seek operation failed: " + std::strerror(errno));

    parse_header_data(parsed_header, header);
    parsed_header.data_offset = offset;
}

size_t NumpyDataReader::read_numpy_data(void* buf, size_t read_size, std::vector<size_t> max_shape) {
    if (!_current_fPtr)
        THROW("Null file pointer");

    // Requested read size bigger than the file size? just read as many bytes as the file size
    read_size = (read_size > _current_file_size) ? _current_file_size : read_size;

    if (std::fseek(_current_fPtr, _curr_file_header.data_offset, SEEK_SET))
        THROW("Seek operation failed: " + std::strerror(errno));

    auto shape = _curr_file_header.shape();
    auto num_dims = max_shape.size();
    std::vector<unsigned> strides(num_dims + 1);
    strides[num_dims] = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
        strides[i] = strides[i + 1] * max_shape[i];
    }

    size_t actual_read_size = 0;
    if (_curr_file_header.type() == RocalTensorDataType::UINT8)
        actual_read_size = parse_numpy_data<u_int8_t>((u_int8_t*)buf, strides, shape);
    else if (_curr_file_header.type() == RocalTensorDataType::UINT32)
        actual_read_size = parse_numpy_data<u_int32_t>((u_int32_t*)buf, strides, shape);
    else if (_curr_file_header.type() == RocalTensorDataType::INT8)
        actual_read_size = parse_numpy_data<int8_t>((int8_t*)buf, strides, shape);
    else if (_curr_file_header.type() == RocalTensorDataType::INT16)
        actual_read_size = parse_numpy_data<int16_t>((int16_t*)buf, strides, shape);
    else if (_curr_file_header.type() == RocalTensorDataType::INT32)
        actual_read_size = parse_numpy_data<int32_t>((int32_t*)buf, strides, shape);
    else if (_curr_file_header.type() == RocalTensorDataType::FP16)
#if defined(AMD_FP16_SUPPORT)
        actual_read_size = parse_numpy_data<half>((half*)buf, strides, shape);
#else
        THROW("FLOAT16 type tensor not supported")
#endif
    else if (_curr_file_header.type() == RocalTensorDataType::FP32)
        actual_read_size = parse_numpy_data<float>((float*)buf, strides, shape);

    return actual_read_size;
}

template <typename T>
size_t NumpyDataReader::parse_numpy_data(T* buf, std::vector<unsigned> strides, std::vector<unsigned> shapes, unsigned dim) {
    if (dim == (shapes.size() - 1)) {
        auto actual_read_size = std::fread(buf, sizeof(T), shapes[dim], _current_fPtr);
        return actual_read_size;
    }
    T* startPtr = buf;
    size_t read_size = 0;
    for (unsigned d = 0; d < shapes[dim]; d++) {
        read_size += parse_numpy_data<T>(startPtr, strides, shapes, dim + 1);
        startPtr += strides[dim + 1];
    }
    return read_size;
}

const NumpyHeaderData NumpyDataReader::get_numpy_header_data() {
    return _curr_file_header;
}

size_t NumpyDataReader::read_data(unsigned char* buf, size_t read_size) {
    if (!_current_fPtr)
        return 0;

    // Requested read size bigger than the file size? just read as many bytes as the file size
    read_size = (read_size > _current_file_size) ? _current_file_size : read_size;

    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
    return actual_read_size;
}

int NumpyDataReader::close() {
    return release();
}

NumpyDataReader::~NumpyDataReader() {
    release();
}

int NumpyDataReader::release() {
    if (!_current_fPtr)
        return 0;
    fclose(_current_fPtr);
    _current_fPtr = nullptr;
    return 0;
}

void NumpyDataReader::reset() {
    if (_shuffle) {
        std::mt19937 rng(_seed);
        std::shuffle(_file_names.begin() + _shard_start_idx_vector[_shard_id],
                     _file_names.begin() + _shard_start_idx_vector[_shard_id] + actual_shard_size_without_padding(), rng);
    }

    if (_stick_to_shard == false)  // Pick elements from the next shard - hence increment shard_id
        increment_shard_id();      // Should work for both single and multiple shards

    _read_counter = 0;

    if (_sharding_info.last_batch_policy == RocalBatchPolicy::DROP) {  // Skipping the dropped batch in next epoch
        for (uint i = 0; i < _batch_size; i++)
            increment_curr_file_idx(_file_names.size());
    }
}

Reader::Status NumpyDataReader::generate_file_names() {
    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("NumpyDataReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

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
                    }
                }
            }
        }
    } else if (!_files.empty()) {
        for (unsigned file_count = 0; file_count < _files.size(); file_count++) {
            std::string file_path = _files[file_count];
            filesys::path pathObj(file_path);
            if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
                // ignore files with extensions .tar, .zip, .7z
                auto file_extension_idx = file_path.find_last_of(".");
                if (file_extension_idx != std::string::npos) {
                    std::string file_extension = file_path.substr(file_extension_idx + 1);
                    std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    if (file_extension != "npy")
                        continue;
                    else {
                        _last_file_name = file_path;
                        _file_names.push_back(file_path);
                        _file_count_all_shards++;
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
                    if (file_extension != "npy")
                        continue;
                }
                ret = open_folder();
                break;  // assume directory has only files.
            } else if (filesys::exists(pathObj) && filesys::is_directory(pathObj)) {
                _folder_path = subfolder_path;
                if (open_folder() != Reader::Status::OK)
                    WRN("NumpyDataReader ShardID [" + TOSTR(_shard_id) + "] File reader cannot access the storage at " + _folder_path);
            }
        }
    }

    if (_file_names.empty())
        ERR("NumpyDataReader ShardID [" + TOSTR(_shard_id) + "] Did not load any file from " + _folder_path)

    size_t padded_samples = ((_shard_size > 0) ? _shard_size : largest_shard_size_without_padding()) % _batch_size;
    _last_batch_padded_size = ((_batch_size > 1) && (padded_samples > 0)) ? (_batch_size - padded_samples) : 0;

    // Pad the _file_names with last element of the shard in the vector when _pad_last_batch_repeated is True
    if (_pad_last_batch_repeated == true) {
        update_filenames_with_padding(_file_names, _batch_size);
    }

    _last_file_name = _file_names[_file_names.size() - 1];
    compute_start_and_end_idx_of_all_shards();

    return ret;
}

Reader::Status NumpyDataReader::subfolder_reading() {
    auto ret = generate_file_names();
    if (!_file_names.empty())
        LOG("NumpyDataReader ShardID [" + TOSTR(_shard_id) + "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + STR(_folder_path))
    return ret;
}

Reader::Status NumpyDataReader::open_folder() {
    if ((_src_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("NumpyDataReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

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
            if (file_extension != "npy")
                continue;
        }
        if (!_meta_data_reader || _meta_data_reader->exists(filename)) {  // Check if the file is present in metadata reader and add to file names list, to avoid issues while lookup
            _file_names.push_back(file_path);
            _last_file_name = file_path;
            _file_count_all_shards++;
        } else {
            WRN("Skipping file," + filename + " as it is not present in metadata reader")
        }
    }

    if (_file_names.empty())
        ERR("NumpyDataReader ShardID [" + TOSTR(_shard_id) + "] Did not load any file from " + _folder_path)
    closedir(_src_dir);
    return Reader::Status::OK;
}

std::string NumpyDataReader::get_root_folder_path() {
    return _folder_path;
}

std::vector<std::string> NumpyDataReader::get_file_paths_from_meta_data_reader() {
    if (_meta_data_reader) {
        return _meta_data_reader->get_relative_file_path();
    } else {
        std::cout << "\n Meta Data Reader is not initialized!";
        return {};
    }
}
