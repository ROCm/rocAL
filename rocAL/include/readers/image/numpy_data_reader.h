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

#pragma once
#include <dirent.h>

#include <memory>
#include <string>
#include <vector>

#include "commons.h"
#include "image_reader.h"
#include "timing_debug.h"

class NumpyDataReader : public Reader {
   public:
    //! Looks up the folder which contains the files, amd loads the image names
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig desc) override;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded images
     \return Size of the loaded resource
    */
    size_t read_data(unsigned char* buf, size_t max_size) override;
    //! Opens the next file in the folder
    /*!
     \return The size of the next file, 0 if couldn't access it
    */
    size_t open() override;

    const NumpyHeaderData get_numpy_header_data() override;

    size_t read_numpy_data(void* buf, size_t read_size, std::vector<size_t> max_shape) override;

    //! Resets the object's state to read from the first file in the folder
    void reset() override;

    //! Returns the name of the latest file opened
    std::string id() override { return _last_id; };

    unsigned count_items() override;

    ~NumpyDataReader() override;

    int close() override;

    NumpyDataReader();

   private:
    //! opens the folder containnig the images
    Reader::Status open_folder();
    Reader::Status subfolder_reading();
    std::string _folder_path;
    DIR* _src_dir;
    DIR* _sub_dir;
    struct dirent* _entity;
    std::vector<std::string> _file_names;
    std::vector<std::string> _files;
    std::vector<NumpyHeaderData> _file_headers;
    unsigned _curr_file_idx;
    FILE* _current_fPtr;
    unsigned _current_file_size;
    std::string _last_id;
    std::string _last_file_name;
    size_t _shard_id = 0;
    size_t _shard_count = 1;  // equivalent of batch size
    //!< _batch_count Defines the quantum count of the images to be read. It's usually equal to the user's batch size.
    /// The loader will repeat images if necessary to be able to have images available in multiples of the load_batch_count,
    /// for instance if there are 10 images in the dataset and _batch_count is 3, the loader repeats 2 images as if there are 12 images available.
    size_t _batch_count = 1;
    size_t _file_id = 0;
    size_t _in_batch_read_count = 0;
    bool _loop;
    bool _shuffle;
    int _read_counter = 0;
    unsigned _seed = 0;
    //!< _file_count_all_shards total_number of files in to figure out the max_batch_size (usually needed for distributed training).
    size_t _file_count_all_shards;
    std::mutex _cache_mutex_;
    std::map<std::string, NumpyHeaderData> _header_cache_;
    const RocalTensorDataType TypeFromNumpyStr(const std::string& format);
    inline void SkipSpaces(const char*& ptr);
    void ParseHeaderContents(NumpyHeaderData& target, const std::string& header);
    template <size_t N>
    void Skip(const char*& ptr, const char (&what)[N]);
    template <size_t N>
    bool TrySkip(const char*& ptr, const char (&what)[N]);
    template <size_t N>
    void SkipFieldName(const char*& ptr, const char (&name)[N]);
    template <typename T = int64_t>
    T ParseInteger(const char*& ptr);
    std::string ParseStringValue(const char*& input, char delim_start = '\'', char delim_end = '\'');
    void ParseHeader(NumpyHeaderData& parsed_header, std::string file_path);
    template <typename T>
    size_t ParseNumpyData(T* buf, std::vector<unsigned> strides, std::vector<unsigned> shapes, unsigned dim = 0);
    bool GetFromCache(const std::string& file_name, NumpyHeaderData& target);
    void UpdateCache(const std::string& file_name, const NumpyHeaderData& value);   
    void incremenet_read_ptr();
    int release();
    size_t get_file_shard_id();
    void incremenet_file_id() { _file_id++; }
    void replicate_last_image_to_fill_last_shard();
    void replicate_last_batch_to_pad_partial_shard();
    TimingDBG _shuffle_time;
};
