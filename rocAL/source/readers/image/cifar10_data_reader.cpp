/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <cerrno>
#include "pipeline/commons.h"
#include <cstring>
#include <algorithm>
#include <random>
#include "readers/image/cifar10_data_reader.h"
#include "readers/file_source_reader.h"
#include "pipeline/filesystem.h"

#if ENABLE_HIP && ENABLE_HIPFILE
#include <cstdlib>
#include <fcntl.h>
#include <strings.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>
#include <hipfile.h>
#endif

#if ENABLE_HIP && ENABLE_HIPFILE
namespace {
constexpr size_t kHipFileAlignment = 4096;
constexpr size_t kHipFileMaxIoSize = 0x7ffff000LU;

inline size_t align_down(size_t value, size_t alignment) {
    return value & ~(alignment - 1);
}

inline size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

inline bool is_device_ptr(const void* ptr) {
    hipPointerAttribute_t attrs{};
    auto status = hipPointerGetAttributes(&attrs, ptr);
    return (status == hipSuccess) && (attrs.type == hipMemoryTypeDevice);
}

inline bool hipfile_forced_compat_mode() {
    const char* v = std::getenv("HIPFILE_FORCE_COMPAT_MODE");
    return v && (!strcasecmp(v, "true"));
}
}  // namespace
#endif

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

Reader::Status CIFAR10DataReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _folder_path = desc.path();
    _batch_size = desc.get_batch_size();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _loop = desc.loop();
    _file_name_prefix = desc.file_prefix();
    _sharding_info = desc.get_sharding_info();
    _pad_last_batch_repeated = _sharding_info.pad_last_batch_repeated;
    _stick_to_shard = _sharding_info.stick_to_shard;
    _shard_size = _sharding_info.shard_size;
    _shuffle = desc.shuffle();
    ret = subfolder_reading();
    // shuffle dataset if set
    if (ret == Reader::Status::OK && _shuffle) {
        std::mt19937 rng1(_shard_id);
        auto rng2 = rng1;
        auto rng3 = rng1;
        std::shuffle(_file_names.begin() + _shard_start_idx_vector[_shard_id],
                            _file_names.begin() + _shard_end_idx_vector[_shard_id], rng1);
        std::shuffle(_file_offsets.begin() + _shard_start_idx_vector[_shard_id],
                            _file_offsets.begin() + _shard_end_idx_vector[_shard_id], rng2);
        std::shuffle(_file_idx.begin() + _shard_start_idx_vector[_shard_id],
                            _file_idx.begin() + _shard_end_idx_vector[_shard_id], rng3);
    }
    return ret;

}

void CIFAR10DataReader::incremenet_read_ptr() {
    _read_counter++;
    increment_curr_file_idx(_file_names.size());
}

size_t CIFAR10DataReader::open() {
    auto file_path = _file_names[_curr_file_idx];  // Get next file name
    auto file_offset = _file_offsets[_curr_file_idx];
    _last_file_idx = _file_idx[_curr_file_idx];
    _last_file_offset = file_offset;
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
#if ENABLE_HIP && ENABLE_HIPFILE
        close_hipfile();
#endif
        _current_fPtr = fopen(file_path.c_str(), "rb");  // Open the file,
        if (!_current_fPtr) {
            return 0;
        }
        _last_file_name = file_path;
        fseek(_current_fPtr, 0, SEEK_END);  // Take the file read pointer to the end
        _total_file_size = ftell(_current_fPtr);
        fseek(_current_fPtr, 0, SEEK_SET);  // Take the file read pointer to the beginning
    }

    if (!_current_fPtr)  // Check if it is ready for reading
        return 0;

    const size_t file_offset_sz = static_cast<size_t>(file_offset);
    if (file_offset_sz >= _total_file_size || _raw_file_size > (_total_file_size - file_offset_sz)) {  // not enough data in the file to read
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

#if ENABLE_HIP && ENABLE_HIPFILE
    if (is_device_ptr(buf)) {
        // If hipFile is forced into compatibility mode, using it here would add extra copies.
        if (!hipfile_forced_compat_mode() && ensure_hipfile_open()) {
            const size_t file_size = static_cast<size_t>(_total_file_size);
            const size_t data_offset = static_cast<size_t>(_last_file_offset) + 1;
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
                            WRN("hipFileRead failed in CIFAR10DataReader::read_data for " + _last_file_name +
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

                            // Near EOF, the AIS path may not support short reads; read the remaining tail via host.
                            const size_t tail_offset = data_offset + direct_bytes;
                            const size_t tail_bytes = read_size - direct_bytes;
                            if (_host_staging.size() < tail_bytes) {
                                _host_staging.resize(tail_bytes);
                            }
                            if (std::fseek(_current_fPtr, static_cast<long>(tail_offset), SEEK_SET)) {
                                WRN("Seek operation failed in CIFAR10DataReader::read_data for " + _last_file_name + ": " +
                                    std::strerror(errno))
                                return direct_bytes;
                            }
                            const size_t host_read_size =
                                std::fread(_host_staging.data(), sizeof(unsigned char), tail_bytes, _current_fPtr);
                            if (host_read_size == 0) {
                                return direct_bytes;
                            }
                            hip_status =
                                hipMemcpy(buf + direct_bytes, _host_staging.data(), host_read_size, hipMemcpyHostToDevice);
                            if (hip_status != hipSuccess) {
                                WRN("hipMemcpyHostToDevice failed in CIFAR10DataReader::read_data tail copy: " + TOSTR(hip_status))
                                return direct_bytes;
                            }
                            return direct_bytes + host_read_size;
                        }
                        WRN("hipMemcpyDeviceToDevice failed in CIFAR10DataReader::read_data: " + TOSTR(hip_status))
                    }
                }
            }
        }

        // Fallback: read to host and copy to device
        if (_host_staging.size() < read_size) {
            _host_staging.resize(read_size);
        }
        size_t host_read_size = fread(_host_staging.data(), sizeof(unsigned char), read_size, _current_fPtr);
        if (host_read_size == 0) {
            return 0;
        }
        auto hip_status = hipMemcpy(buf, _host_staging.data(), host_read_size, hipMemcpyHostToDevice);
        if (hip_status != hipSuccess) {
            WRN("hipMemcpyHostToDevice failed in CIFAR10DataReader::read_data: " + TOSTR(hip_status))
            return 0;
        }
        return host_read_size;
    }
#endif

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
#if ENABLE_HIP && ENABLE_HIPFILE
    close_hipfile();
    if (_hipfile_scratch) {
        hipPointerAttribute_t attrs{};
        if (hipPointerGetAttributes(&attrs, _hipfile_scratch) == hipSuccess) {
            (void)hipSetDevice(attrs.device);
        }
        if (_hipfile_scratch_registered) {
            (void)hipFileBufDeregister(_hipfile_scratch);
            _hipfile_scratch_registered = false;
        }
        (void)hipFree(_hipfile_scratch);
        _hipfile_scratch = nullptr;
        _hipfile_scratch_size = 0;
    }
#endif
}

int CIFAR10DataReader::release() {
    // do not need to close file here since data is read from the same file continuously
    return 0;
}

void CIFAR10DataReader::reset() {
    if (_shuffle) {
        std::mt19937 rng1(_shard_id);
        auto rng2 = rng1;
        auto rng3 = rng1;
        std::shuffle(_file_names.begin() + _shard_start_idx_vector[_shard_id],
                            _file_names.begin() + _shard_start_idx_vector[_shard_id] + actual_shard_size_without_padding(), rng1);
        std::shuffle(_file_offsets.begin() + _shard_start_idx_vector[_shard_id],
                            _file_offsets.begin() + _shard_start_idx_vector[_shard_id] + actual_shard_size_without_padding(), rng2);
        std::shuffle(_file_idx.begin() + _shard_start_idx_vector[_shard_id],
                            _file_idx.begin() + _shard_start_idx_vector[_shard_id] + actual_shard_size_without_padding(), rng3);
    }
    if (_stick_to_shard == false)  // Pick elements from the next shard - hence increment shard_id
        increment_shard_id();      // Should work for both single and multiple shards
    _read_counter = 0;
    if (_sharding_info.last_batch_policy == RocalBatchPolicy::DROP) {  // Skipping the dropped batch in next epoch
        for (uint32_t i = 0; i < _batch_size; i++)
            increment_curr_file_idx(_file_names.size());
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
    size_t padded_samples = 0;
    // Pad the _file_names with last element of the shard in the vector when _pad_last_batch_repeated is True
    padded_samples = ((_shard_size > 0) ? _shard_size : largest_shard_size_without_padding()) % _batch_size;
    _last_batch_padded_size = ((_batch_size > 1) && (padded_samples > 0)) ? (_batch_size - padded_samples) : 0;

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
            auto start_offset = _file_offsets.begin() + start_idx + total_padded_samples;
            auto end_offset = start_offset + actual_shard_size_without_padding;
            auto start_file_idx = _file_idx.begin() + start_idx + total_padded_samples;
            auto end_file_idx = start_file_idx + actual_shard_size_without_padding;
            if (largest_shard_size % _batch_size) {
                size_t num_padded_samples = 0;
                num_padded_samples = (largest_shard_size - actual_shard_size_without_padding) + _batch_size - (largest_shard_size % _batch_size);
                _file_count_all_shards += num_padded_samples;
                _file_names.insert(end, num_padded_samples, _file_names[start_idx + actual_shard_size_without_padding + total_padded_samples - 1]);
                _file_offsets.insert(end_offset, num_padded_samples, _file_offsets[start_idx + actual_shard_size_without_padding + total_padded_samples - 1]);
                _file_idx.insert(end_file_idx, num_padded_samples, _file_idx[start_idx + actual_shard_size_without_padding + total_padded_samples - 1]);
                total_padded_samples += num_padded_samples;
            }
        }
    }
    compute_start_and_end_idx_of_all_shards();
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
            if (!fp) {
                WRN("CIFAR10DataReader:: Could not open file " + file_path)
                continue;
            }
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

#if ENABLE_HIP && ENABLE_HIPFILE
void CIFAR10DataReader::close_hipfile() {
    if (_hipfile_handle) {
        hipFileHandleDeregister(reinterpret_cast<hipFileHandle_t>(_hipfile_handle));
        _hipfile_handle = nullptr;
    }
    if (_hipfile_fd >= 0) {
        ::close(_hipfile_fd);
        _hipfile_fd = -1;
    }
    _hipfile_open_path.clear();
}

bool CIFAR10DataReader::ensure_hipfile_open() {
    if (_hipfile_handle && _hipfile_open_path == _last_file_name) {
        return true;
    }

    close_hipfile();
    if (_last_file_name.empty()) {
        return false;
    }

    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_DIRECT
    flags |= O_DIRECT;
#endif
    _hipfile_fd = ::open(_last_file_name.c_str(), flags);
    if (_hipfile_fd < 0) {
        // Fall back to non-O_DIRECT open; hipFile will attempt to open an unbuffered FD internally.
        int fallback_flags = O_RDONLY;
#ifdef O_CLOEXEC
        fallback_flags |= O_CLOEXEC;
#endif
        _hipfile_fd = ::open(_last_file_name.c_str(), fallback_flags);
        if (_hipfile_fd < 0) {
            WRN("Failed to open file for hipFile: " + _last_file_name + " (" + std::strerror(errno) + ")")
            return false;
        }
    }

    hipFileDescr_t descr{};
    descr.type = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = _hipfile_fd;

    hipFileHandle_t handle = nullptr;
    auto err = hipFileHandleRegister(&handle, &descr);
    if (err.err != hipFileSuccess) {
        WRN("hipFileHandleRegister failed for " + _last_file_name + ": " + std::string(hipFileGetOpErrorString(err.err)))
        ::close(_hipfile_fd);
        _hipfile_fd = -1;
        return false;
    }

    _hipfile_handle = handle;
    _hipfile_open_path = _last_file_name;
    return true;
}

bool CIFAR10DataReader::ensure_hipfile_scratch(size_t size_in_bytes) {
    if (size_in_bytes == 0) {
        return false;
    }
    if (_hipfile_scratch && _hipfile_scratch_size >= size_in_bytes) {
        return true;
    }

    if (_hipfile_scratch) {
        if (_hipfile_scratch_registered) {
            (void)hipFileBufDeregister(_hipfile_scratch);
            _hipfile_scratch_registered = false;
        }
        (void)hipFree(_hipfile_scratch);
        _hipfile_scratch = nullptr;
        _hipfile_scratch_size = 0;
    }

    auto hip_status = hipMalloc(&_hipfile_scratch, size_in_bytes);
    if (hip_status != hipSuccess || !_hipfile_scratch) {
        WRN("hipMalloc failed for CIFAR10 hipFile scratch buffer: " + TOSTR(hip_status))
        return false;
    }
    _hipfile_scratch_size = size_in_bytes;

    auto hipfile_err = hipFileBufRegister(_hipfile_scratch, _hipfile_scratch_size, 0);
    _hipfile_scratch_registered = (hipfile_err.err == hipFileSuccess);
    return true;
}
#endif
