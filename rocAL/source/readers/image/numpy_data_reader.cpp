/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include "numpy_data_reader.h"

#include <commons.h>

#include <algorithm>
#include <boost/filesystem.hpp>
#include <cassert>

namespace filesys = boost::filesystem;

NumpyDataReader::NumpyDataReader() : _shuffle_time("shuffle_time", DBG_TIMING) {
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

unsigned NumpyDataReader::count_items() {
    if (_loop)
        return _file_names.size();

    int ret = ((int)_file_names.size() - _read_counter);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status NumpyDataReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _file_id = 0;
    _folder_path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    ret = subfolder_reading();
    // the following code is required to make every shard the same size:: required for multi-gpu training
    if (_shard_count > 1 && _batch_count > 1) {
        int _num_batches = _file_names.size() / _batch_count;
        int max_batches_per_shard = (_file_count_all_shards + _shard_count - 1) / _shard_count;
        max_batches_per_shard = (max_batches_per_shard + _batch_count - 1) / _batch_count;
        if (_num_batches < max_batches_per_shard) {
            replicate_last_batch_to_pad_partial_shard();
        }
    }
    // shuffle dataset if set
    _shuffle_time.start();
    if (ret == Reader::Status::OK && _shuffle)
        std::random_shuffle(_file_names.begin(), _file_names.end());
    _shuffle_time.end();
    return ret;
}

void NumpyDataReader::incremenet_read_ptr() {
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
}

size_t NumpyDataReader::open() {
    auto file_path = _file_names[_curr_file_idx];  // Get next file name
    incremenet_read_ptr();
    _last_id = file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        _last_id.erase(0, last_slash_idx + 1);
    }

    ParseHeader(_file_headers[_curr_file_idx], file_path);
    fseek(_current_fPtr, 0, SEEK_SET);  // Take the file pointer back to the start

    return _file_headers[_curr_file_idx].nbytes();
}

const RocalTensorDataType NumpyDataReader::TypeFromNumpyStr(const std::string& format) {
    if (format == "u1") return RocalTensorDataType::UINT8;
    // if (format == "u2") return TypeTable::GetTypeInfo<uint16_t>();   // Currently not supported in rocAL
    if (format == "u4") return RocalTensorDataType::UINT32;
    // if (format == "u8") return TypeTable::GetTypeInfo<uint64_t>();   // Currently not supported in rocAL
    if (format == "i1") return RocalTensorDataType::INT8;
    // if (format == "i2") return TypeTable::GetTypeInfo<int16_t>();    // Currently not supported in rocAL
    if (format == "i4") return RocalTensorDataType::INT32;
    // if (format == "i8") return TypeTable::GetTypeInfo<int64_t>();    // Currently not supported in rocAL
    if (format == "f2")
#if defined(AMD_FP16_SUPPORT)
        return RocalTensorDataType::FP16;
#else
        THROW("FLOAT16 type tensor not supported")
#endif
    if (format == "f4") return RocalTensorDataType::FP32;
    // if (format == "f8") return TypeTable::GetTypeInfo<double>();     // Currently not supported in rocAL
    THROW("Unknown Numpy type string");
}

inline void NumpyDataReader::SkipSpaces(const char*& ptr) {
    while (::isspace(*ptr))
        ptr++;
}

template <size_t N>
void NumpyDataReader::Skip(const char*& ptr, const char (&what)[N]) {
    if (strncmp(ptr, what, N - 1))
        THROW("Found wrong symbol during parsing");
    ptr += N - 1;
}

template <size_t N>
bool NumpyDataReader::TrySkip(const char*& ptr, const char (&what)[N]) {
    if (!strncmp(ptr, what, N - 1)) {
        ptr += N - 1;
        return true;
    } else {
        return false;
    }
}

template <size_t N>
void NumpyDataReader::SkipFieldName(const char*& ptr, const char (&name)[N]) {
    SkipSpaces(ptr);
    Skip(ptr, "'");
    Skip(ptr, name);
    Skip(ptr, "'");
    SkipSpaces(ptr);
    Skip(ptr, ":");
    SkipSpaces(ptr);
}

template <typename T = int64_t>
T NumpyDataReader::ParseInteger(const char*& ptr) {
    char* out_ptr = const_cast<char*>(ptr);  // strtol takes a non-const pointer
    T value = static_cast<T>(strtol(ptr, &out_ptr, 10));
    if (out_ptr == ptr)
        THROW("Parse error: expected a number.");
    ptr = out_ptr;
    return value;
}

std::string NumpyDataReader::ParseStringValue(const char*& input, char delim_start, char delim_end) {
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

void NumpyDataReader::ParseHeaderContents(NumpyHeaderData& target, const std::string& header) {
    const char* hdr = header.c_str();
    SkipSpaces(hdr);
    Skip(hdr, "{");
    SkipFieldName(hdr, "descr");
    auto typestr = ParseStringValue(hdr);
    // < means LE, | means N/A, = means native. In all those cases, we can read
    bool little_endian = (typestr[0] == '<' || typestr[0] == '|' || typestr[0] == '=');
    if (!little_endian)
        THROW("Big Endian files are not supported.");
    target._type_info = TypeFromNumpyStr(typestr.substr(1));

    SkipSpaces(hdr);
    Skip(hdr, ",");
    SkipFieldName(hdr, "fortran_order");
    if (TrySkip(hdr, "True")) {
        target._fortran_order = true;
    } else if (TrySkip(hdr, "False")) {
        target._fortran_order = false;
    } else {
        THROW("Failed to parse fortran_order field.");
    }
    SkipSpaces(hdr);
    Skip(hdr, ",");
    SkipFieldName(hdr, "shape");
    Skip(hdr, "(");
    SkipSpaces(hdr);
    target._shape.clear();
    while (*hdr != ')') {
        // ParseInteger already skips the leading spaces (strtol does).
        target._shape.push_back(static_cast<unsigned>(ParseInteger<int64_t>(hdr)));
        SkipSpaces(hdr);
        if (!(TrySkip(hdr, ",")) && (target._shape.size() <= 1))
            THROW("The first number in a tuple must be followed by a comma.");
    }
    if (target._fortran_order) {
        // cheapest thing to do is to define the tensor in an reversed way
        std::reverse(target._shape.begin(), target._shape.end());
    }
}

void NumpyDataReader::ParseHeader(NumpyHeaderData& parsed_header, std::string file_path) {
    // check if the file is actually a numpy file
    std::vector<char> token(128);
    _current_fPtr = std::fopen(file_path.c_str(), "rb");
    if (_current_fPtr == nullptr)
        THROW("Could not open file " + file_path + ": " + std::strerror(errno));
    int64_t n_read = std::fread(token.data(), 1, 10, _current_fPtr);
    if (n_read != 10)
        THROW("Can not read header.");
    token[n_read] = '\0';

    // check if heqder is too short
    std::string header = std::string(token.data());
    if (header.find_first_of("NUMPY") == std::string::npos)
        THROW("File is not a numpy file.");

    // extract header length
    uint16_t header_len = 0;
    memcpy(&header_len, &token[8], 2);
    if ((header_len + 10) % 16 != 0)
        THROW("Error extracting header length.");

    // read header: the offset is a magic number
    int64_t offset = 6 + 1 + 1 + 2;
    // the header_len can be 4GiB according to the NPYv2 file format
    // specification: https://numpy.org/neps/nep-0001-npy-format.html
    // while this allocation could be sizable, it is performed on the host.
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

    ParseHeaderContents(parsed_header, header);
    parsed_header._data_offset = offset;
}

size_t NumpyDataReader::read_numpy_data(void* buf, size_t read_size, std::vector<size_t> max_shape) {
    if (!_current_fPtr)
        THROW("Null file pointer");

    // Requested read size bigger than the file size? just read as many bytes as the file size
    read_size = (read_size > _current_file_size) ? _current_file_size : read_size;

    if (std::fseek(_current_fPtr, _file_headers[_curr_file_idx]._data_offset, SEEK_SET))
        THROW("Seek operation failed: " + std::strerror(errno));

    auto shape = _file_headers[_curr_file_idx].shape();
    auto num_dims = max_shape.size();
    std::vector<unsigned> strides(num_dims + 1);
    strides[num_dims] = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
        strides[i] = strides[i + 1] * max_shape[i];
    }

    size_t actual_read_size = 0;
    if (_file_headers[_curr_file_idx].type() == RocalTensorDataType::UINT8)
        actual_read_size = ParseNumpyData<u_int8_t>((u_int8_t*)buf, strides, shape);
    if (_file_headers[_curr_file_idx].type() == RocalTensorDataType::UINT32)
        actual_read_size = ParseNumpyData<u_int32_t>((u_int32_t*)buf, strides, shape);
    if (_file_headers[_curr_file_idx].type() == RocalTensorDataType::INT8)
        actual_read_size = ParseNumpyData<int8_t>((int8_t*)buf, strides, shape);
    if (_file_headers[_curr_file_idx].type() == RocalTensorDataType::INT32)
        actual_read_size = ParseNumpyData<int32_t>((int32_t*)buf, strides, shape);
    if (_file_headers[_curr_file_idx].type() == RocalTensorDataType::FP16)
#if defined(AMD_FP16_SUPPORT)
        actual_read_size = ParseNumpyData<half>((half*)buf, strides, shape);
#else
        THROW("FLOAT16 type tensor not supported")
#endif
    if (_file_headers[_curr_file_idx].type() == RocalTensorDataType::FP32)
        actual_read_size = ParseNumpyData<float>((float*)buf, strides, shape);

    return actual_read_size;
}

template <typename T>
size_t NumpyDataReader::ParseNumpyData(T* buf, std::vector<unsigned> strides, std::vector<unsigned> shapes, unsigned dim) {
    if (dim == (shapes.size() - 1)) {
        auto actual_read_size = std::fread(buf, sizeof(T), shapes[dim], _current_fPtr);
        return actual_read_size;
    }
    T* startPtr = buf;
    size_t read_size = 0;
    for (unsigned d = 0; d < shapes[dim]; d++) {
        read_size += ParseNumpyData<T>(startPtr, strides, shapes, dim + 1);
        startPtr += strides[dim + 1];
    }
    return read_size;
}

const NumpyHeaderData NumpyDataReader::get_numpy_header_data() {
    return _file_headers[_curr_file_idx];
}

size_t NumpyDataReader::read_data(unsigned char* buf, size_t read_size) {
    if (!_current_fPtr)
        return 0;

    // Requested read size bigger than the file size? just read as many bytes as the file size
    read_size = (read_size > _current_file_size) ? _current_file_size : read_size;

    if (std::fseek(_current_fPtr, _file_headers[_curr_file_idx]._data_offset, SEEK_SET))
        THROW("Seek operation failed: " + std::strerror(errno));

    size_t actual_read_size = std::fread(buf, 1, _file_headers[_curr_file_idx].nbytes(), _current_fPtr);
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
    _shuffle_time.start();
    if (_shuffle) std::random_shuffle(_file_names.begin(), _file_names.end());
    _shuffle_time.end();
    _read_counter = 0;
    _curr_file_idx = 0;
}

Reader::Status NumpyDataReader::subfolder_reading() {
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
    for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
        std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
        filesys::path pathObj(subfolder_path);
        if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
            // ignore files with extensions .tar, .zip, .7z
            auto file_extension_idx = subfolder_path.find_last_of(".");
            if (file_extension_idx != std::string::npos) {
                std::string file_extension = subfolder_path.substr(file_extension_idx + 1);
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
    if (_in_batch_read_count > 0 && _in_batch_read_count < _batch_count) {
        replicate_last_image_to_fill_last_shard();
        LOG("NumpyDataReader ShardID [" + TOSTR(_shard_id) + "] Replicated " + _folder_path + _last_file_name + " " + TOSTR((_batch_count - _in_batch_read_count)) + " times to fill the last batch")
    }
    _file_headers.resize(_file_names.size());
    if (!_file_names.empty())
        LOG("NumpyDataReader ShardID [" + TOSTR(_shard_id) + "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path)
    return ret;
}

void NumpyDataReader::replicate_last_image_to_fill_last_shard() {
    for (size_t i = _in_batch_read_count; i < _batch_count; i++)
        _file_names.push_back(_last_file_name);
}

void NumpyDataReader::replicate_last_batch_to_pad_partial_shard() {
    if (_file_names.size() >= _batch_count) {
        for (size_t i = 0; i < _batch_count; i++)
            _file_names.push_back(_file_names[i - _batch_count]);
    }
}

Reader::Status NumpyDataReader::open_folder() {
    if ((_src_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("NumpyDataReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        if (_entity->d_type != DT_REG)
            continue;

        if (get_file_shard_id() != _shard_id) {
            _file_count_all_shards++;
            incremenet_file_id();
            continue;
        }
        _in_batch_read_count++;
        _in_batch_read_count = (_in_batch_read_count % _batch_count == 0) ? 0 : _in_batch_read_count;
        std::string file_path = _folder_path;
        file_path.append("/");
        file_path.append(_entity->d_name);
        _last_file_name = file_path;
        _file_names.push_back(file_path);
        _file_count_all_shards++;
        incremenet_file_id();
    }
    if (_file_names.empty())
        WRN("NumpyDataReader ShardID [" + TOSTR(_shard_id) + "] Did not load any file from " + _folder_path)

    closedir(_src_dir);
    return Reader::Status::OK;
}

size_t NumpyDataReader::get_file_shard_id() {
    if (_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    // return (_file_id / (_batch_count)) % _shard_count;
    return _file_id % _shard_count;
}
