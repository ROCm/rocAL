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

#pragma once
#include <dirent.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "pipeline/commons.h"
#include "pipeline/timing_debug.h"
#include "readers/image/image_reader.h"

class NumpyDataReader : public Reader {
   public:
    //! Looks up the folder which contains the files, amd loads the numpy files
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig desc) override;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded numpy data
     \return Size of the loaded resource
    */
    size_t read_data(unsigned char* buf, size_t max_size) override;
    //! Opens the next file in the folder
    /*!
     \return The size of the next file, 0 if couldn't access it
    */
    size_t open() override;

    //! Returns the numpy header data 
    /*!
     \return The numpy header data of the current file
    */
    const NumpyHeaderData get_numpy_header_data() override;

    //! Reads the next numpy file
    /*!
     \param buf User's provided buffer to receive the loaded numpy data
     \param raad_size Requested read size by the user
     \param buf Max dimension strides of the output tensor used for padding along the dimensions
     \return Size of the loaded resource
    */
    size_t read_numpy_data(void* buf, size_t read_size, std::vector<unsigned>& max_shape) override;

    //! Resets the object's state to read from the first file in the folder
    void reset() override;

    //! Returns the name of the latest file opened
    std::string id() override { return _last_id; };

    //! Returns the name of the latest file_path opened
    const std::string file_path() override { return _last_file_path; }

    unsigned count_items() override;

    ~NumpyDataReader() override;

    int close() override;

    NumpyDataReader();

    std::string get_root_folder_path() override;  // Returns the root folder path

   private:
    //! opens the folder containing the numpy arrays
    Reader::Status open_folder();
    Reader::Status subfolder_reading();
    std::string _folder_path;
    std::string _file_list_path;
    DIR *_src_dir;
    DIR *_sub_dir;
    struct dirent *_entity;
    std::vector<std::string> _file_names;
    std::vector<std::string> _files;
    std::vector<NumpyHeaderData> _file_headers;
    FILE *_current_fPtr;
    unsigned _current_file_size;
    NumpyHeaderData _curr_file_header;
    std::string _last_id;
    std::string _last_file_name, _last_file_path, _absolute_file_path;
    size_t _batch_size = 1;
    bool _loop;
    bool _shuffle;
    int _read_counter = 0;
    unsigned _seed = 0;
    //!< _file_count_all_shards total_number of files in to figure out the max_batch_size (usually needed for distributed training).
    std::mutex _cache_mutex;
    std::map<std::string, NumpyHeaderData> _header_cache;
    const RocalTensorDataType get_dtype(const std::string& format);
    inline void skip_spaces(const char*& ptr);
    void parse_header_data(NumpyHeaderData& target, const std::string& header);
    template <size_t N>
    void skip_char(const char*& ptr, const char (&what)[N]);
    template <size_t N>
    bool try_skip_char(const char*& ptr, const char (&what)[N]);
    template <size_t N>
    void skip_field(const char*& ptr, const char (&name)[N]);
    template <typename T = int64_t>
    T parse_int(const char*& ptr);
    std::string parse_string(const char*& input, char delim_start = '\'', char delim_end = '\'');
    void parse_header(NumpyHeaderData& parsed_header, std::string file_path);
    template <typename T>
    size_t parse_numpy_data(T* buf, std::vector<unsigned>& strides, std::vector<unsigned>& shapes, unsigned dim = 0);
    bool get_header_from_cache(const std::string& file_name, NumpyHeaderData& target);
    void update_header_cache(const std::string& file_name, const NumpyHeaderData& value);
    void incremenet_read_ptr();
    int release();
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    Reader::Status generate_file_names();            // Function that would generate _file_names containing all the samples in the dataset
};
