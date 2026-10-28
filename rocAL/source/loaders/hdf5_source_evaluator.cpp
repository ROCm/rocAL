/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "loaders/hdf5_source_evaluator.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <set>
#include <limits>

#include "pipeline/exception.h"
#include "pipeline/filesystem.h"

#if defined(ROCAL_HDF5)
#include <H5Cpp.h>
#endif

std::mutex& hdf5_api_mutex() {
    static std::mutex mutex;
    return mutex;
}

namespace {

bool is_hdf5_path(const filesys::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension == ".h5" || extension == ".hdf5";
}

std::vector<std::string> resolve_hdf5_files(const std::string& source_path,
                                            const std::vector<std::string>& requested_files) {
    if (source_path.empty())
        THROW("HDF5 reader requires a non-empty file_root")

    const filesys::path root(source_path);
    if (!filesys::exists(root) || !filesys::is_directory(root))
        THROW("HDF5 reader file_root is not an accessible directory: " + source_path)

    std::vector<std::string> resolved_files;
    if (requested_files.empty()) {
        for (const auto& entry : filesys::directory_iterator(root)) {
            const auto path = entry.path();
            if (filesys::is_regular_file(path) && is_hdf5_path(path))
                resolved_files.push_back(filesys::canonical(path).string());
        }
    } else {
        resolved_files.reserve(requested_files.size());
        for (const auto& requested_file : requested_files) {
            if (requested_file.empty())
                THROW("HDF5 reader files cannot contain an empty path")
            filesys::path path(requested_file);
            if (!path.is_absolute())
                path = root / path;
            if (!filesys::exists(path) || !filesys::is_regular_file(path))
                THROW("HDF5 reader cannot access file: " + path.string())
            if (!is_hdf5_path(path))
                THROW("HDF5 reader only accepts .h5 and .hdf5 files: " + path.string())
            resolved_files.push_back(filesys::canonical(path).string());
        }
    }

    if (requested_files.empty())
        std::sort(resolved_files.begin(), resolved_files.end());
    if (std::set<std::string>(resolved_files.begin(), resolved_files.end()).size() != resolved_files.size())
        THROW("HDF5 reader files contains duplicate paths")
    if (resolved_files.empty())
        THROW("HDF5 reader did not find any .h5 or .hdf5 files in: " + source_path)
    return resolved_files;
}

void validate_dataset_keys(const std::vector<std::string>& dataset_keys) {
    if (dataset_keys.empty())
        THROW("HDF5 reader requires at least one dataset key")

    std::set<std::string> unique_keys;
    for (const auto& key : dataset_keys) {
        if (key.empty())
            THROW("HDF5 reader dataset_keys cannot contain an empty key")
        if (!unique_keys.insert(key).second)
            THROW("HDF5 reader dataset_keys contains duplicate key: " + key)
    }
}

#if defined(ROCAL_HDF5)
RocalTensorDataType get_rocal_data_type(const H5::DataType& hdf5_type,
                                        const std::string& file,
                                        const std::string& key) {
    const auto type_class = hdf5_type.getClass();
    const auto type_size = hdf5_type.getSize();

    if (type_class == H5T_FLOAT && type_size == sizeof(float))
        return RocalTensorDataType::FP32;

    if (type_class == H5T_INTEGER) {
        H5::IntType integer_type(hdf5_type.getId());
        const bool is_signed = integer_type.getSign() == H5T_SGN_2;
        if (is_signed && type_size == sizeof(int16_t))
            return RocalTensorDataType::INT16;
        if (is_signed && type_size == sizeof(int32_t))
            return RocalTensorDataType::INT32;
        if (!is_signed && type_size == sizeof(uint8_t))
            return RocalTensorDataType::UINT8;
        if (!is_signed && type_size == sizeof(uint32_t))
            return RocalTensorDataType::UINT32;
    }

    THROW("HDF5 reader does not support the data type for key '" + key +
          "' in file: " + file)
}

std::vector<size_t> get_dataset_shape(const H5::DataSet& dataset,
                                      const std::string& file,
                                      const std::string& key) {
    const auto data_space = dataset.getSpace();
    const int rank = data_space.getSimpleExtentNdims();
    if (rank <= 0)
        THROW("HDF5 reader requires a non-scalar dataset for key '" + key +
              "' in file: " + file)

    std::vector<hsize_t> hdf5_dimensions(static_cast<size_t>(rank));
    data_space.getSimpleExtentDims(hdf5_dimensions.data());

    std::vector<size_t> shape;
    shape.reserve(hdf5_dimensions.size());
    for (const auto dimension : hdf5_dimensions) {
        if (dimension == 0 || dimension > std::numeric_limits<uint32_t>::max())
            THROW("HDF5 reader found an empty or oversized dimension for key '" + key +
                  "' in file: " + file)
        shape.push_back(static_cast<size_t>(dimension));
    }
    return shape;
}
#endif

}  // namespace

Hdf5SourceInfo Hdf5SourceEvaluator::evaluate(
    const std::string& source_path,
    const std::vector<std::string>& files,
    const std::vector<std::string>& dataset_keys) const {
    validate_dataset_keys(dataset_keys);
    auto resolved_files = resolve_hdf5_files(source_path, files);

#if !defined(ROCAL_HDF5)
    (void)resolved_files;
    THROW("rocAL was built without HDF5 reader support")
#else
    std::vector<Hdf5DatasetInfo> dataset_info;
    dataset_info.reserve(dataset_keys.size());

    try {
        for (size_t file_index = 0; file_index < resolved_files.size(); ++file_index) {
            std::lock_guard<std::mutex> lock(hdf5_api_mutex());
            H5::Exception::dontPrint();
            const auto& file_path = resolved_files[file_index];
            H5::H5File input_file(file_path, H5F_ACC_RDONLY);

            for (size_t key_index = 0; key_index < dataset_keys.size(); ++key_index) {
                const auto& key = dataset_keys[key_index];
                if (H5Lexists(input_file.getId(), key.c_str(), H5P_DEFAULT) <= 0)
                    THROW("HDF5 reader cannot find dataset key '" + key +
                          "' in file: " + file_path)

                const auto dataset = input_file.openDataSet(key);
                const auto shape = get_dataset_shape(dataset, file_path, key);
                const auto data_type = get_rocal_data_type(dataset.getDataType(), file_path, key);

                if (file_index == 0) {
                    dataset_info.push_back({key, shape, data_type});
                    continue;
                }

                auto& expected = dataset_info[key_index];
                if (shape.size() != expected.max_shape.size())
                    THROW("HDF5 reader dataset rank changed for key '" + key +
                          "' in file: " + file_path)
                if (data_type != expected.data_type)
                    THROW("HDF5 reader dataset data type changed for key '" + key +
                          "' in file: " + file_path)
                if (shape != expected.max_shape)
                    THROW("HDF5 reader dataset shape changed for key '" + key +
                          "' in file: " + file_path)
            }
        }
    } catch (const H5::Exception& error) {
        THROW("HDF5 reader schema inspection failed: " + error.getDetailMsg())
    }

    return {std::move(resolved_files), std::move(dataset_info)};
#endif
}
