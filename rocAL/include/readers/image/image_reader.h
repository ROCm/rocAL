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

#pragma once
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <lmdb.h>
#include "meta_data/meta_data_reader.h"
#include "readers/video/video_properties.h"

#define CHECK_LMDB_RETURN_STATUS(status)                                                          \
    do {                                                                                          \
        if (status != MDB_SUCCESS)                                                                \
            THROW("LMDB error, " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + \
                  #status + ":" + std::string(mdb_strerror(status)));                             \
    } while (0)

enum class StorageType {
    FILE_SYSTEM = 0,
    TF_RECORD = 1,
    UNCOMPRESSED_BINARY_DATA = 2,  // experimental: added for supporting cifar10 data set
    CAFFE_LMDB_RECORD = 3,
    CAFFE2_LMDB_RECORD = 4,
    COCO_FILE_SYSTEM = 5,
    SEQUENCE_FILE_SYSTEM = 6,
    MXNET_RECORDIO = 7,
    VIDEO_FILE_SYSTEM = 8,
    EXTERNAL_FILE_SOURCE = 9,      // to support reading from external source
};

enum class ExternalSourceFileMode {
    FILENAME = 0,
    RAWDATA_COMPRESSED = 1,
    RAWDATA_UNCOMPRESSED = 2,
    NONE = 3,
};

struct ReaderConfig {
    explicit ReaderConfig(StorageType type, std::string path = "", std::string json_path = "",
                          const std::map<std::string, std::string> feature_key_map = std::map<std::string, std::string>(),
                          bool shuffle = false, bool loop = false) : _type(type), _path(path), _json_path(json_path), _feature_key_map(feature_key_map), _shuffle(shuffle), _loop(loop) {}
    virtual StorageType type() { return _type; };
    void set_path(const std::string &path) { _path = path; }
    void set_shard_id(size_t shard_id) { _shard_id = shard_id; }
    void set_shard_count(size_t shard_count) { _shard_count = shard_count; }
    void set_cpu_num_threads(size_t cpu_num_threads) { _cpu_num_threads = cpu_num_threads; }
    void set_json_path(const std::string &json_path) { _json_path = json_path; }
    /// \param read_batch_count Tells the reader it needs to read the images in multiples of load_batch_count. If available images not divisible to load_batch_count,
    /// the reader will repeat images to make available images an even multiple of this load_batch_count
    void set_batch_count(size_t read_batch_count) { _batch_count = read_batch_count; }
    /// \param loop if True the reader's available images still the same no matter how many images have been read
    bool shuffle() { return _shuffle; }
    bool loop() { return _loop; }
    void set_shuffle(bool shuffle) { _shuffle = shuffle; }
    void set_loop(bool loop) { _loop = loop; }
    void set_meta_data_reader(std::shared_ptr<MetaDataReader> meta_data_reader) { _meta_data_reader = meta_data_reader; }
    void set_sequence_length(unsigned sequence_length) { _sequence_length = sequence_length; }
    void set_frame_step(unsigned step) { _sequence_frame_step = step; }
    void set_frame_stride(unsigned stride) { _sequence_frame_stride = stride; }
    void set_external_filemode(ExternalSourceFileMode mode) { _file_mode = mode; }
    void set_last_batch_policy(std::pair<RocalBatchPolicy, bool> last_batch_info) {
        _last_batch_info = last_batch_info;
    }
    size_t get_shard_count() { return _shard_count; }
    size_t get_shard_id() { return _shard_id; }
    size_t get_cpu_num_threads() { return _cpu_num_threads; }
    size_t get_batch_size() { return _batch_count; }
    size_t get_sequence_length() { return _sequence_length; }
    size_t get_frame_step() { return _sequence_frame_step; }
    size_t get_frame_stride() { return _sequence_frame_stride; }
    std::string path() { return _path; }
#ifdef ROCAL_VIDEO
    void set_video_properties(VideoProperties video_prop) { _video_prop = video_prop; }
    VideoProperties get_video_properties() { return _video_prop; }
#endif
    std::string json_path() { return _json_path; }
    std::map<std::string, std::string> feature_key_map() { return _feature_key_map; }
    void set_file_prefix(const std::string &prefix) { _file_prefix = prefix; }
    std::string file_prefix() { return _file_prefix; }
    std::shared_ptr<MetaDataReader> meta_data_reader() { return _meta_data_reader; }
    ExternalSourceFileMode mode() { return _file_mode; }
    std::pair<RocalBatchPolicy, bool> get_last_batch_policy() { return _last_batch_info; }

   private:
    StorageType _type = StorageType::FILE_SYSTEM;
    std::string _path = "";
    std::string _json_path = "";
    std::map<std::string, std::string> _feature_key_map;
    size_t _shard_count = 1;
    size_t _shard_id = 0;
    size_t _cpu_num_threads = 1;
    size_t _batch_count = 1;      //!< The reader will repeat images if necessary to be able to have images in multiples of the _batch_count.
    size_t _sequence_length = 1;  // Video reader module sequence length
    size_t _sequence_frame_step;
    size_t _sequence_frame_stride = 1;
    bool _shuffle = false;
    bool _loop = false;
    std::string _file_prefix = "";  //!< to read only files with prefix. supported only for cifar10_data_reader and tf_record_reader
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    ExternalSourceFileMode _file_mode = ExternalSourceFileMode::NONE;
    std::pair<RocalBatchPolicy, bool> _last_batch_info = {RocalBatchPolicy::FILL, true};
#ifdef ROCAL_VIDEO
    VideoProperties _video_prop;
#endif
};

// MXNet image recordio struct - used to read the contents from the MXNet recordIO files.
struct ImageRecordIOHeader {
    uint32_t flag;        // flag of the header
    float label;          // label field that returns label of images
    uint64_t image_id[2]; /* unique image index
                           *  image_id[1] is always set to 0,
                           *  reserved for future purposes for 128bit id
                           *  image_id[0] is used to store image id
                           */
};


class Reader {
   public:
    enum class Status {
        OK = 0
    };

    // TODO: change method names to open_next, read_next , ...

    //! Initializes the resource which it's spec is defined by the desc argument
    /*!
     \param desc description of the resource infor. It's exact fields are defind by the derived class.
     \return status of the being able to locate the resource pointed to by the desc
    */
    virtual Status initialize(ReaderConfig desc) = 0;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded items
     \return Size of the loaded resource
    */

    //! Opens the next item and returns it's size
    /*!
     \return Size of the item, if 0 failed to access it
    */
    virtual size_t open() = 0;

    //! Copies the data of the opened item to the buf
    virtual size_t read_data(unsigned char *buf, size_t read_size) = 0;

    //! Closes the opened item
    virtual int close() = 0;

    //! Starts reading from the first item in the resource
    virtual void reset() = 0;

    //! Returns the name/identifier of the last item opened in this resource
    virtual std::string id() = 0;
    //! Returns the number of items remained in this resource

     //! Returns the path of the last item opened in this resource
    virtual const std::string file_path() { THROW("File path is not set by the reader") }

    virtual unsigned count_items() = 0;

    virtual ~Reader() = default;

    //! Returns the number of images in the last batch
    virtual size_t last_batch_padded_size() { return 0; }
};
