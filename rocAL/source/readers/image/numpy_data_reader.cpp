/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <math.h>
#include "pipeline/commons.h"
#include "pipeline/filesystem.h"
#include "readers/image/numpy_data_reader.h"

#if ENABLE_HIP && ENABLE_HIPFILE
#include <cstdint>
#include <cstdlib>
#include <strings.h>
#include <sys/stat.h>
#include <hip/hip_runtime_api.h>
#include <hipfile.h>
#endif

// the HEADER_OFFSET is a magic number - the first 10 bytes store info about numpy version and header len
// The first 6 bytes are a magic string: exactly \x93NUMPY.
// The next 1 byte is an unsigned byte: the major version number of the file format, e.g. \x01.
// The next 1 byte is an unsigned byte: the minor version number of the file format, e.g. \x00
// The next 2 bytes form a little-endian unsigned short int: the length of the header data HEADER_LEN.
#define HEADER_OFFSET 10
#define CALL_AND_CHECK_FLAG(func)     \
    {                                 \
        func;                         \
        if (_header_parsing_failed) { \
            return;                   \
        }                             \
    }

#define CHECK_CONDITION_AND_SET_FLAG(condition, error_msg, return_value) \
    {                                                                    \
        if (condition) {                                                 \
            ERR(error_msg)                                               \
            _header_parsing_failed = true;                               \
            return return_value;                                         \
        }                                                                \
    }

#if ENABLE_HIP && ENABLE_HIPFILE
namespace {
/*
 * hipFile Integration Notes:
 * - hipFile GPU Direct I/O is disabled by default. Users must set ROCAL_USE_HIPFILE=1 to enable.
 * - hipFile enables direct-to-GPU reads (GPU Direct Storage) from NVMe to GPU HBM, bypassing
 *   the host staging path (fread -> pinned host memory -> zero-copy GPU mapping).
 * - The AMD fastpath requires 4KB-aligned file offsets, buffer offsets, and IO sizes.
 * - NumPy payload offsets are typically not 4KB-aligned (data_offset is 64-128 bytes for NPY v1),
 *   so we read an aligned window into a reusable device scratch buffer registered with
 *   hipFileBufRegister, then D2D copy the requested subrange into the final output.
 * - If HIPFILE_FORCE_COMPAT_MODE=true, hipFile would internally add extra copies, so we bypass it.
 * - The feature uses fileno() on the existing FILE* rather than a separate O_DIRECT open,
 *   avoiding the overhead of a second open() syscall per file.
 * - The scratch buffer is allocated once and reused across all file reads. For uniform-shape
 *   datasets (common case), the aligned read size is the same for every file.
 * - This feature is most beneficial when the dataset is too large to fit in the OS page cache.
 *   When data fits in cache, fread() serves from RAM, while hipFile reads from NVMe.
 *   hipFile wins for large (TB-scale) datasets or cold-storage scenarios.
 */

// hipFile fastpath uses 4KB/page alignment when statx dio-align data is unavailable.
constexpr size_t kHipFileAlignment = 4096;
// hipFileRead() max transfer size (~2GiB - 4KiB) from hipFile's MAX_RW_COUNT.
constexpr size_t kHipFileMaxIoSize = 0x7ffff000LU;

inline size_t align_down(size_t value, size_t alignment) {
    return value & ~(alignment - 1);
}

inline size_t align_up(size_t value, size_t alignment) {
    const size_t mask = alignment - 1;
    if (value > (std::numeric_limits<size_t>::max() - mask)) {
        return std::numeric_limits<size_t>::max() & ~mask;
    }
    return (value + mask) & ~mask;
}

inline bool hipfile_forced_compat_mode() {
    static const bool cached = [] {
        const char* v = std::getenv("HIPFILE_FORCE_COMPAT_MODE");
        return v && (!strcasecmp(v, "true"));
    }();
    return cached;
}

// hipFile is disabled by default. Users must set ROCAL_USE_HIPFILE=1 to opt in.
inline bool hipfile_enabled() {
    static const bool cached = [] {
        const char* v = std::getenv("ROCAL_USE_HIPFILE");
        return v && (std::string(v) == "1");
    }();
    return cached;
}
}  // namespace
#endif

NumpyDataReader::NumpyDataReader() {
    _loop = false;
    _shuffle = false;
    _file_count_all_shards = 0;
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

    _header_parsing_failed = false;
    auto ret = get_header_from_cache(file_path, _curr_file_header);
    if (!ret) {
        parse_header(_curr_file_header, file_path);
        if(_header_parsing_failed) {
            ERR("Numpy header parsing failed");
            return 0;
        }
        update_header_cache(file_path, _curr_file_header);
    } else {
        _current_file_ptr = std::fopen(file_path.c_str(), "rb");
        if (_current_file_ptr == nullptr) {
            ERR("Could not open file " + file_path + ": " + std::strerror(errno));
            return 0;
        }
    }
    fseek(_current_file_ptr, 0, SEEK_SET);  // Take the file pointer back to the start

#if ENABLE_HIP && ENABLE_HIPFILE
    close_hipfile();
#endif
    return _curr_file_header.numpy_data_nbytes();  // Returns the numpy array data size (in bytes)
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

template <size_t N>
void NumpyDataReader::skip_char(const char*& ptr, const char (&what)[N]) {
    CHECK_CONDITION_AND_SET_FLAG(strncmp(ptr, what, N - 1), "Found wrong symbol during parsing, expected symbol: " + std::string(what), void())
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
    while (std::isspace(*ptr)) ptr++;
    CALL_AND_CHECK_FLAG(skip_char(ptr, "'"));
    CALL_AND_CHECK_FLAG(skip_char(ptr, name));
    CALL_AND_CHECK_FLAG(skip_char(ptr, "'"));
    while (std::isspace(*ptr)) ptr++;
    CALL_AND_CHECK_FLAG(skip_char(ptr, ":"));
    while (std::isspace(*ptr)) ptr++;
}

template <typename T>
T NumpyDataReader::parse_int(const char*& ptr) {
    char* out_ptr = const_cast<char*>(ptr);  // strtol takes a non-const pointer
    T value = static_cast<T>(strtol(ptr, &out_ptr, 10));
    CHECK_CONDITION_AND_SET_FLAG(out_ptr == ptr, "Parse error: expected a number.", value)
    ptr = out_ptr;
    return value;
}

std::string NumpyDataReader::parse_string(const char*& input, char delim_start, char delim_end) {
    CHECK_CONDITION_AND_SET_FLAG(*input++ != delim_start, "Expected \'" + std::to_string(delim_start) + "\'", "")
    std::string out;
    for (; *input != '\0'; input++) {
        if (*input == '\\') {
            char c = *++input;
            if ((c == '\\') || (c == '\'') || (c == '\t') || (c == '\n') || (c == '\"')) {
                out += c;
            } else {
                out += '\\';
                out += *input;
            }
        } else if (*input == delim_end) {
            break;
        } else {
            out += *input;
        }
    }
    CHECK_CONDITION_AND_SET_FLAG(*input++ != delim_end, "Expected \'" + std::to_string(delim_end) + "\'", "")
    return out;
}

void NumpyDataReader::parse_header_data(NumpyHeaderData& target, const std::string& header) {
    const char* hdr = header.c_str();
    while (std::isspace(*hdr)) hdr++;
    CALL_AND_CHECK_FLAG(skip_char(hdr, "{"));
    CALL_AND_CHECK_FLAG(skip_field(hdr, "descr"));
    auto typestr = parse_string(hdr);
    if (_header_parsing_failed) return;
    // < means LE, | means N/A, = means native. In all those cases, we can read
    bool little_endian = (typestr[0] == '<' || typestr[0] == '|' || typestr[0] == '=');
    CHECK_CONDITION_AND_SET_FLAG(!little_endian, "Big Endian files are not supported.", void())
    CHECK_CONDITION_AND_SET_FLAG(_numpy_str_to_rocal_dtype.find(typestr.substr(1)) == _numpy_str_to_rocal_dtype.end(), typestr.substr(1) + " numpy dtype not supported in rocAL", void())
    target.type_info = _numpy_str_to_rocal_dtype[typestr.substr(1)];
    while (std::isspace(*hdr)) hdr++;
    CALL_AND_CHECK_FLAG(skip_char(hdr, ","));
    CALL_AND_CHECK_FLAG(skip_field(hdr, "fortran_order"));
    if (try_skip_char(hdr, "True")) {
        target.fortran_order = true;
    } else if (try_skip_char(hdr, "False")) {
        target.fortran_order = false;
    } else {
        CHECK_CONDITION_AND_SET_FLAG(true, "Failed to parse fortran_order field.", void())
    }
    while (std::isspace(*hdr)) hdr++;
    CALL_AND_CHECK_FLAG(skip_char(hdr, ","));
    CALL_AND_CHECK_FLAG(skip_field(hdr, "shape"));
    CALL_AND_CHECK_FLAG(skip_char(hdr, "("));
    while (std::isspace(*hdr)) hdr++;
    target.array_shape.clear();
    while (*hdr != ')') {
        // parse_int already skips the leading spaces (strtol does).
        auto shape = parse_int<int64_t>(hdr);
        if(_header_parsing_failed) return;
        target.array_shape.push_back(static_cast<unsigned>(shape));
        while (std::isspace(*hdr)) hdr++;
        CHECK_CONDITION_AND_SET_FLAG(!(try_skip_char(hdr, ",")) && (target.array_shape.size() <= 1), "The first number in a tuple must be followed by a comma.", void())
    }
    if (target.fortran_order) {
        // cheapest thing to do is to define the tensor in an reversed way
        std::reverse(target.array_shape.begin(), target.array_shape.end());
    }
}

void NumpyDataReader::parse_header(NumpyHeaderData& parsed_header, std::string file_path) {
    std::vector<char> token(HEADER_OFFSET + 1);  // Need to store 10 bytes of numpy header info and null termination character
    _current_file_ptr = std::fopen(file_path.c_str(), "rb");
    CHECK_CONDITION_AND_SET_FLAG(_current_file_ptr == nullptr, "Could not open file " + file_path + ": " + std::strerror(errno), void())

    int64_t offset = HEADER_OFFSET;
    int64_t n_read = std::fread(token.data(), 1, offset, _current_file_ptr);
    // check if header is too short
    CHECK_CONDITION_AND_SET_FLAG(n_read != offset, "Can not read numpy header file contents", void())
    token[n_read] = '\0';

    // rocAL only supports numpy V1 headers
    // https://numpy.org/neps/nep-0001-npy-format.html
    int np_api_version = token[6];
    CHECK_CONDITION_AND_SET_FLAG(np_api_version != 1, "rocAL only supports reading npy files with NPY file format version 1", void())
    // check if the file is actually a numpy file
    std::string header = std::string(token.data());
    CHECK_CONDITION_AND_SET_FLAG(header.find("NUMPY") == std::string::npos, "File is not a numpy file", void())

    // extract header length which can have up to 65535 bytes - NPYv1 format
    uint16_t header_len = 0;
    memcpy(&header_len, &token[8], 2);
    CHECK_CONDITION_AND_SET_FLAG((header_len + 10) % 16 != 0, "Error extracting numpy header length", void())

    token.resize(header_len + 1);
    CHECK_CONDITION_AND_SET_FLAG(std::fseek(_current_file_ptr, offset, SEEK_SET), "Seek operation failed in " + file_path + ": " + std::strerror(errno), void())
    n_read = std::fread(token.data(), 1, header_len, _current_file_ptr);
    CHECK_CONDITION_AND_SET_FLAG(n_read != header_len, "Can not read numpy header upto header_len", void())    
    token[header_len] = '\0';
    header = std::string(token.data());
    CHECK_CONDITION_AND_SET_FLAG(header.find('{') == std::string::npos, "Header is corrupted", void())
    offset += header_len;
    CHECK_CONDITION_AND_SET_FLAG(std::fseek(_current_file_ptr, offset, SEEK_SET), "Seek operation failed in " + file_path + ": " + std::strerror(errno), void())

    parse_header_data(parsed_header, header);
    if (_header_parsing_failed) return;
    parsed_header.data_offset = offset;
}

size_t NumpyDataReader::read_numpy_data(void* buf, size_t read_size, std::vector<unsigned>& strides_in_dims) {
    if (!_current_file_ptr) {
        ERR("Null file pointer");
        return 0;
    }
    
    auto shape = _curr_file_header.shape();
    auto data_type_size = tensor_data_size(_curr_file_header.type());

#if ENABLE_HIP && ENABLE_HIPFILE
    if (!_output_is_device_initialized) {
        hipPointerAttribute_t attrs{};
        auto status = hipPointerGetAttributes(&attrs, buf);
        _output_is_device = (status == hipSuccess) && (attrs.type == hipMemoryTypeDevice);
        _output_is_device_initialized = true;
    }

    // hipFile GPU Direct I/O: only active when ROCAL_USE_HIPFILE=1, output buffer is device
    // memory, and HIPFILE_FORCE_COMPAT_MODE is not set. Falls back to the standard fread() path
    // when any condition is not met.
    if (_output_is_device && hipfile_enabled()) {
        // hipFile path currently supports only contiguous output layouts
        if (!hipfile_forced_compat_mode() && strides_in_dims[0] == _curr_file_header.size() && ensure_hipfile_open() && _hipfile_file_size > 0 &&
            _curr_file_header.data_offset >= 0 && static_cast<size_t>(_curr_file_header.data_offset) <= _hipfile_file_size &&
            read_size <= _hipfile_file_size - static_cast<size_t>(_curr_file_header.data_offset)) {
            const size_t file_size = _hipfile_file_size;
            const size_t data_offset = static_cast<size_t>(_curr_file_header.data_offset);
            const size_t desired_end = data_offset + read_size;

            const size_t aligned_file_offset = align_down(data_offset, kHipFileAlignment);
            const size_t max_aligned_end = align_down(file_size, kHipFileAlignment);
            size_t aligned_end = align_up(desired_end, kHipFileAlignment);
            if (aligned_end > max_aligned_end) {
                aligned_end = max_aligned_end;
            }

            if (aligned_end > aligned_file_offset) {
                const size_t io_size = aligned_end - aligned_file_offset;
                const size_t offset_in_scratch = data_offset - aligned_file_offset;
                const size_t direct_bytes = std::min(read_size, aligned_end - data_offset);

                if (ensure_hipfile_scratch(io_size)) {
                    bool io_ok = true;
                    for (size_t chunk_offset = 0; chunk_offset < io_size; chunk_offset += kHipFileMaxIoSize) {
                        const size_t chunk_size = std::min(kHipFileMaxIoSize, io_size - chunk_offset);
                        const ssize_t nread = hipFileRead(reinterpret_cast<hipFileHandle_t>(_hipfile_handle), _hipfile_scratch, chunk_size,
                                                          static_cast<hoff_t>(aligned_file_offset + chunk_offset),
                                                          static_cast<hoff_t>(chunk_offset));
                        if (nread < 0 || static_cast<size_t>(nread) != chunk_size) {
                            ERR("hipFileRead failed in NumpyDataReader::read_numpy_data for " + _last_file_path +
                                " (" + std::string(IS_HIPFILE_ERR(nread) ? HIPFILE_ERRSTR(nread) : std::strerror(errno)) +
                                ", nread=" + std::to_string(nread) + ")")
                            io_ok = false;
                            break;
                        }
                    }

                    if (io_ok) {
                        auto hip_status =
                            hipMemcpy(buf, static_cast<unsigned char*>(_hipfile_scratch) + offset_in_scratch, direct_bytes,
                                      hipMemcpyDeviceToDevice);
                        if (hip_status == hipSuccess) {
                            if (direct_bytes == read_size) {
                                return read_size;
                            }

                            // The hipFile device-direct I/O path may not support short reads; read the remaining tail via host.
                            const size_t tail_offset = data_offset + direct_bytes;
                            const size_t tail_bytes = read_size - direct_bytes;
                            if (_host_staging.size() < tail_bytes) {
                                _host_staging.resize(tail_bytes);
                            }
                            if (std::fseek(_current_file_ptr, static_cast<long>(tail_offset), SEEK_SET) != 0) {
                                ERR("Seek operation failed for " + _last_file_path + ": " + std::strerror(errno));
                                return direct_bytes;
                            }
                            const size_t host_read_size =
                                std::fread(_host_staging.data(), sizeof(unsigned char), tail_bytes, _current_file_ptr);
                            if (host_read_size == 0) {
                                return direct_bytes;
                            }
                            hip_status =
                                hipMemcpy(static_cast<unsigned char*>(buf) + direct_bytes, _host_staging.data(), host_read_size,
                                          hipMemcpyHostToDevice);
                            if (hip_status != hipSuccess) {
                                ERR("hipMemcpyHostToDevice failed in NumpyDataReader::read_numpy_data tail copy: " + TOSTR(hip_status))
                                return direct_bytes;
                            }
                            return direct_bytes + host_read_size;
                        }
                        ERR("hipMemcpyDeviceToDevice failed in NumpyDataReader::read_numpy_data: " + TOSTR(hip_status))
                    }
                }
            }
        }

        // Fallback: read to host and copy to device (supports both contiguous and strided outputs)
        if (data_type_size > 0 && static_cast<size_t>(strides_in_dims[0]) > std::numeric_limits<size_t>::max() / data_type_size) {
            ERR("Overflow computing output_bytes in NumpyDataReader::read_numpy_data for " + _last_file_path)
            return 0;
        }
        const size_t output_bytes = static_cast<size_t>(strides_in_dims[0]) * data_type_size;
        if (_host_staging.size() < output_bytes) {
            _host_staging.resize(output_bytes);
        }
        std::fill(_host_staging.begin(), _host_staging.begin() + output_bytes, 0);

        if (std::fseek(_current_file_ptr, _curr_file_header.data_offset, SEEK_SET)) {
            ERR("Seek operation failed for " + _last_file_path + ": " + std::strerror(errno));
            return 0;
        }
        size_t host_bytes_read = 0;
        if (strides_in_dims[0] == _curr_file_header.size()) {
            host_bytes_read = std::fread(_host_staging.data(), sizeof(unsigned char), read_size, _current_file_ptr);
        } else {
            host_bytes_read = parse_numpy_data(_host_staging.data(), strides_in_dims, shape, data_type_size);
        }
        if (host_bytes_read == 0) {
            return 0;
        }
        auto hip_status = hipMemcpy(buf, _host_staging.data(), output_bytes, hipMemcpyHostToDevice);
        if (hip_status != hipSuccess) {
            ERR("hipMemcpyHostToDevice failed in NumpyDataReader::read_numpy_data: " + TOSTR(hip_status))
            return 0;
        }
        return host_bytes_read;
    }
#endif

    if (std::fseek(_current_file_ptr, _curr_file_header.data_offset, SEEK_SET)) {
        ERR("Seek operation failed for " + _last_file_path + ": " + std::strerror(errno));
        return 0;
    }

    if (strides_in_dims[0] == _curr_file_header.size())
        return std::fread((unsigned char*)buf, sizeof(unsigned char), _curr_file_header.numpy_data_nbytes(), _current_file_ptr);
    else
        return parse_numpy_data((unsigned char*)buf, strides_in_dims, shape, data_type_size);
}

size_t NumpyDataReader::parse_numpy_data(unsigned char* buf, std::vector<unsigned>& strides_in_dims, std::vector<unsigned>& shapes, size_t dtype_size, unsigned dim) {
    if (dim == (shapes.size() - 1)) {
        auto actual_read_size = std::fread(buf, sizeof(unsigned char), shapes[dim] * dtype_size, _current_file_ptr);
        return actual_read_size;
    }
    unsigned char* startPtr = buf;
    size_t read_size = 0;
    for (unsigned d = 0; d < shapes[dim]; d++) {
        read_size += parse_numpy_data(startPtr, strides_in_dims, shapes, dtype_size, dim + 1);
        startPtr += (strides_in_dims[dim + 1] * dtype_size);
    }
    return read_size;
}

const NumpyHeaderData NumpyDataReader::get_numpy_header_data() {
    return _curr_file_header;
}

size_t NumpyDataReader::read_data(unsigned char* buf, size_t read_size) {
    if (!_current_file_ptr)
        return 0;

    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_file_ptr);
    return actual_read_size;
}

int NumpyDataReader::close() {
    return release();
}

NumpyDataReader::~NumpyDataReader() {
    release();
#if ENABLE_HIP && ENABLE_HIPFILE
    if (_hipfile_scratch_alloc) {
        if (_hipfile_scratch_device_id >= 0) {
            (void)hipSetDevice(_hipfile_scratch_device_id);
        }
        if (_hipfile_scratch_registered) {
            (void)hipFileBufDeregister(_hipfile_scratch);
            _hipfile_scratch_registered = false;
        }
        (void)hipFree(_hipfile_scratch_alloc);
        _hipfile_scratch_alloc = nullptr;
        _hipfile_scratch = nullptr;
        _hipfile_scratch_size = 0;
        _hipfile_scratch_device_id = -1;
    }
#endif
}

int NumpyDataReader::release() {
    if (!_current_file_ptr)
        return 0;
#if ENABLE_HIP && ENABLE_HIPFILE
    close_hipfile();
#endif
    fclose(_current_file_ptr);
    _current_file_ptr = nullptr;
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
    std::string full_path = _folder_path;

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
                    if (!filesys::exists(_folder_path)) {
                        ERR(file_path + " is a relative path but root path doesn't exist");
                        continue;
                    }
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
                        if (!filesys::exists(_folder_path)) {
                            ERR("File list contains relative paths but root path doesn't exists");
                            continue;
                        }
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
    } else if (!_files.empty()) { // If the user passes a list of filenames, use them instead of reading from folder
        for (unsigned file_count = 0; file_count < _files.size(); file_count++) {
            std::string file_path = _files[file_count];
            filesys::path pathObj(file_path);
            // ignore files with extensions .tar, .zip, .7z
            auto file_extension_idx = file_path.find_last_of(".");
            if (file_extension_idx != std::string::npos) {
                std::string file_extension = file_path.substr(file_extension_idx + 1);
                std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(),
                                [](unsigned char c) { return std::tolower(c); });
                if (file_extension == "npy") {
                    if (filesys::path(file_path).is_relative()) {  // Only add root path if the file list contains relative file paths
                        if (!filesys::exists(_folder_path)) {
                            ERR(file_path + " is a relative path but root path doesn't exist");
                            continue;
                        }
                        file_path = _folder_path + "/" + file_path;
                    }
                    if (filesys::exists(file_path) && filesys::is_regular_file(file_path)) {
                        _last_file_name = file_path;
                        _file_names.push_back(file_path);
                        _file_count_all_shards++;
                    }
                }
            }
        }
    } else {
        for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
            std::string subfolder_path = full_path + "/" + entry_name_list[dir_count];
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

#if ENABLE_HIP && ENABLE_HIPFILE
void NumpyDataReader::close_hipfile() {
    if (_hipfile_handle) {
        hipFileHandleDeregister(reinterpret_cast<hipFileHandle_t>(_hipfile_handle));
        _hipfile_handle = nullptr;
    }
    _hipfile_open_path.clear();
    _hipfile_file_size = 0;
}

bool NumpyDataReader::ensure_hipfile_open() {
    if (_hipfile_handle && _hipfile_open_path == _last_file_path) {
        return true;
    }

    close_hipfile();
    if (_last_file_path.empty() || !_current_file_ptr) {
        return false;
    }

    const int client_fd = ::fileno(_current_file_ptr);
    if (client_fd < 0) {
        ERR("Failed to get file descriptor for hipFile: " + _last_file_path)
        return false;
    }

    struct stat st {};
    if (::fstat(client_fd, &st) == 0) {
        _hipfile_file_size = static_cast<size_t>(st.st_size);
    } else {
        ERR("fstat failed for hipFile: " + _last_file_path + " (errno " + std::to_string(errno) + "): " + std::string(std::strerror(errno)))
        _hipfile_file_size = 0;
    }

    hipFileDescr_t descr{};
    descr.type = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = client_fd;

    hipFileHandle_t handle = nullptr;
    auto err = hipFileHandleRegister(&handle, &descr);
    if (err.err != hipFileSuccess) {
        ERR("hipFileHandleRegister failed for " + _last_file_path + ": " + std::string(hipFileGetOpErrorString(err.err)))
        _hipfile_file_size = 0;
        return false;
    }

    _hipfile_handle = handle;
    _hipfile_open_path = _last_file_path;
    return true;
}

bool NumpyDataReader::ensure_hipfile_scratch(size_t size_in_bytes) {
    if (size_in_bytes == 0) {
        return false;
    }
    if (_hipfile_scratch && _hipfile_scratch_size >= size_in_bytes) {
        return _hipfile_scratch_registered;
    }

    if (_hipfile_scratch_alloc) {
        if (_hipfile_scratch_registered) {
            (void)hipFileBufDeregister(_hipfile_scratch);
            _hipfile_scratch_registered = false;
        }
        if (_hipfile_scratch_device_id >= 0) {
            (void)hipSetDevice(_hipfile_scratch_device_id);
        }
        (void)hipFree(_hipfile_scratch_alloc);
        _hipfile_scratch_alloc = nullptr;
        _hipfile_scratch = nullptr;
        _hipfile_scratch_size = 0;
        _hipfile_scratch_device_id = -1;
    }

    if (size_in_bytes > std::numeric_limits<size_t>::max() - (kHipFileAlignment - 1)) {
        ERR("Requested Numpy hipFile scratch buffer size is too large and would overflow during alignment")
        return false;
    }
    const size_t alloc_size = size_in_bytes + (kHipFileAlignment - 1);
    auto hip_status = hipMalloc(&_hipfile_scratch_alloc, alloc_size);
    if (hip_status != hipSuccess || !_hipfile_scratch_alloc) {
        ERR("hipMalloc failed for Numpy hipFile scratch buffer: " + TOSTR(hip_status))
        return false;
    }

    const uintptr_t addr = reinterpret_cast<uintptr_t>(_hipfile_scratch_alloc);
    const uintptr_t aligned_addr = (addr + (kHipFileAlignment - 1)) & ~(static_cast<uintptr_t>(kHipFileAlignment - 1));
    _hipfile_scratch = reinterpret_cast<void*>(aligned_addr);
    _hipfile_scratch_size = size_in_bytes;
    int device_id = -1;
    if (hipGetDevice(&device_id) == hipSuccess) {
        _hipfile_scratch_device_id = device_id;
    }

    auto hipfile_err = hipFileBufRegister(_hipfile_scratch, _hipfile_scratch_size, 0);
    _hipfile_scratch_registered = (hipfile_err.err == hipFileSuccess);
    if (!_hipfile_scratch_registered) {
        ERR("hipFileBufRegister failed for Numpy hipFile scratch buffer: " + std::string(hipFileGetOpErrorString(hipfile_err.err)))
        (void)hipFree(_hipfile_scratch_alloc);
        _hipfile_scratch_alloc = nullptr;
        _hipfile_scratch = nullptr;
        _hipfile_scratch_size = 0;
        _hipfile_scratch_device_id = -1;
        return false;
    }
    return true;
}
#endif

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
