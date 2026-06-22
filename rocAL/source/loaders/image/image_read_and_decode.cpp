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

#include "loaders/image/image_read_and_decode.h"

#include <algorithm>
#include <omp.h>
#include <cstring>
#include <iterator>
#include <string>

#include "decoders/image/decoder_factory.h"
#include "readers/image/external_source_reader.h"

namespace {
constexpr size_t kMaxRocJpegDecoderCount = 4;
constexpr size_t kMinRocJpegSubBatchSize = 32;

size_t choose_rocjpeg_decoder_count(size_t batch_size, size_t num_threads) {
    const size_t requested_decoder_count =
        std::max<size_t>(1, std::min(kMaxRocJpegDecoderCount, num_threads));

    // Multiple rocJPEG decoder instances are beneficial only when each decoder
    // receives enough images to amortize OpenMP, rocJPEG, HIP, and resize/setup
    // overhead. Keep small batches on one decoder instead of splitting them
    // into small sub-batches.
    const size_t batch_limited_decoder_count =
        std::max<size_t>(1, batch_size / kMinRocJpegSubBatchSize);

    return std::min(requested_decoder_count, batch_limited_decoder_count);
}
}  // namespace

std::tuple<Decoder::ColorFormat, unsigned>
interpret_color_format(RocalColorFormat color_format) {
    switch (color_format) {
        case RocalColorFormat::RGB24:
            return std::make_tuple(Decoder::ColorFormat::RGB, 3);

        case RocalColorFormat::BGR24:
            return std::make_tuple(Decoder::ColorFormat::BGR, 3);

        case RocalColorFormat::U8:
            return std::make_tuple(Decoder::ColorFormat::GRAY, 1);

        default:
            throw std::invalid_argument("Invalid color format\n");
    }
}

Timing
ImageReadAndDecode::timing() {
    Timing t;
    t.decode_time = _decode_time.get_timing();
    t.read_time = _file_load_time.get_timing();
    return t;
}

ImageReadAndDecode::ImageReadAndDecode() : _file_load_time("FileLoadTime", DBG_TIMING),
                                           _decode_time("DecodeTime", DBG_TIMING) {
}

ImageReadAndDecode::~ImageReadAndDecode() {
    _reader = nullptr;
    _decoder.clear();
}

void ImageReadAndDecode::create(ReaderConfig reader_config, DecoderConfig decoder_config, int batch_size, int device_id) {
    // Can initialize it to any decoder types if needed
    _batch_size = batch_size;
    _num_threads = reader_config.get_cpu_num_threads();
    _compressed_buff.resize(batch_size);
    _decoder.resize(batch_size);
    _actual_read_size.resize(batch_size);
    _image_names.resize(batch_size);
    _compressed_image_size.resize(batch_size);
    _decompressed_buff_ptrs.resize(_batch_size);
    _actual_decoded_width.resize(_batch_size);
    _actual_decoded_height.resize(_batch_size);
    _original_height.resize(_batch_size);
    _original_width.resize(_batch_size);
    _decoder_config = decoder_config;
    _random_crop_dec_param = nullptr;
    _device_id = device_id;
    if (_decoder_config._type == DecoderType::FUSED_TURBO_JPEG || _decoder_config._type == DecoderType::ROCJPEG_CROPPED) {
        auto random_aspect_ratio = decoder_config.get_random_aspect_ratio();
        auto random_area = decoder_config.get_random_area();
        AspectRatioRange aspect_ratio_range = std::make_pair((float)random_aspect_ratio[0], (float)random_aspect_ratio[1]);
        AreaRange area_range = std::make_pair((float)random_area[0], (float)random_area[1]);
        _random_crop_dec_param = std::make_shared<RocalRandomCropDecParam>(aspect_ratio_range, area_range, decoder_config.get_num_attempts(), _batch_size);
    }
    if ((_decoder_config._type != DecoderType::SKIP_DECODE)) {
        if (_decoder_config._type == DecoderType::ROCJPEG || _decoder_config._type == DecoderType::ROCJPEG_CROPPED) {
            for (int i = 0; i < batch_size; i++) {
                _compressed_buff[i].resize(MAX_COMPRESSED_SIZE);  // If we don't need MAX_COMPRESSED_SIZE we can remove this & resize in load module
            }
            const size_t rocjpeg_decoder_count = choose_rocjpeg_decoder_count(static_cast<size_t>(batch_size), _num_threads);
            _rocjpeg_decoders.resize(rocjpeg_decoder_count);
            _rocjpeg_sub_batch_sizes.resize(rocjpeg_decoder_count);

            const size_t base_sub_batch = static_cast<size_t>(batch_size) / rocjpeg_decoder_count;
            const size_t sub_batch_remainder = static_cast<size_t>(batch_size) % rocjpeg_decoder_count;
            for (size_t decoder_index = 0; decoder_index < rocjpeg_decoder_count; decoder_index++) {
                const size_t sub_batch_size = base_sub_batch + ((decoder_index < sub_batch_remainder) ? 1 : 0);
                _rocjpeg_sub_batch_sizes[decoder_index] = sub_batch_size;
                _rocjpeg_decoders[decoder_index] = create_decoder(decoder_config);
                _rocjpeg_decoders[decoder_index]->initialize(device_id, sub_batch_size);
            }
        } else {
            for (int i = 0; i < batch_size; i++) {
                _compressed_buff[i].resize(MAX_COMPRESSED_SIZE);  // If we don't need MAX_COMPRESSED_SIZE we can remove this & resize in load module
                _decoder[i] = create_decoder(decoder_config);
                _decoder[i]->initialize(device_id);
            }
        }
    }
    _reader = create_reader(reader_config);
    _is_external_source = (reader_config.type() == StorageType::EXTERNAL_FILE_SOURCE);
}

void ImageReadAndDecode::feed_external_input(const std::vector<std::string>& input_images_names, const std::vector<unsigned char *>& input_buffer,
                                             const std::vector<ROIxywh>& roi_xywh,
                                             unsigned int max_width, unsigned int max_height, unsigned int channels, ExternalSourceFileMode mode, bool eos) {
    std::vector<size_t> image_size;
    std::vector<unsigned> image_roi_w, image_roi_h;
    image_size.reserve(roi_xywh.size());
    image_roi_w.resize(roi_xywh.size());
    image_roi_h.resize(roi_xywh.size());
    size_t max_image_size = max_width * max_height * channels;
    for (unsigned int i = 0; i < roi_xywh.size(); i++) {
        if (mode == ExternalSourceFileMode::RAWDATA_UNCOMPRESSED) {
            image_size[i] = max_image_size;
            image_roi_w[i] = roi_xywh[i].w;
            image_roi_h[i] = roi_xywh[i].h;
        }
        else if (mode == ExternalSourceFileMode::RAWDATA_COMPRESSED)
            image_size[i] = roi_xywh[i].h;
    }
    auto ext_reader = std::static_pointer_cast<ExternalSourceReader>(_reader);
    if (mode == ExternalSourceFileMode::FILENAME)
        ext_reader->feed_file_names(input_images_names, input_images_names.size(), eos);
    else if (mode == ExternalSourceFileMode::RAWDATA_COMPRESSED)
        ext_reader->feed_data(input_buffer, image_size, mode, eos, {}, {}, max_width, max_height, channels);
    else if (mode == ExternalSourceFileMode::RAWDATA_UNCOMPRESSED)
        ext_reader->feed_data(input_buffer, image_size, mode, eos, image_roi_w, image_roi_h, max_width, max_height, channels);
}

void ImageReadAndDecode::reset() {
    // TODO: Reload images from the folder if needed
    _reader->reset();
    _set_device_id = false;
}

size_t
ImageReadAndDecode::count() {
    return _reader->count_items();
}

void ImageReadAndDecode::set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader) {
    _randombboxcrop_meta_data_reader = randombboxcrop_meta_data_reader;
}

std::vector<std::vector<float>>&
ImageReadAndDecode::get_batch_random_bbox_crop_coords() {
    // Return the crop co-ordinates for a batch of images
    return _crop_coords_batch;
}

void ImageReadAndDecode::set_batch_random_bbox_crop_coords(std::vector<std::vector<float>> crop_coords) {
    _crop_coords_batch = crop_coords;
}

size_t
ImageReadAndDecode::last_batch_padded_size() {
    return _reader->last_batch_padded_size();
}

LoaderModuleStatus
ImageReadAndDecode::load(unsigned char *buff,
                         std::vector<std::string> &names,
                         const size_t max_decoded_width,
                         const size_t max_decoded_height,
                         std::vector<uint32_t> &roi_width,
                         std::vector<uint32_t> &roi_height,
                         std::vector<uint32_t> &actual_width,
                         std::vector<uint32_t> &actual_height,
                         RocalColorFormat output_color_format,
                         bool decoder_keep_original) {
    if (max_decoded_width == 0 || max_decoded_height == 0)
        THROW("Zero image dimension is not valid")
    if (!buff)
        THROW("Null pointer passed as output buffer")
    if (_reader->count_items() < _batch_size)
        return LoaderModuleStatus::NO_MORE_DATA_TO_READ;
    // load images/frames from the disk and push them as a large image onto the buff
    unsigned file_counter = 0;
    const auto ret = interpret_color_format(output_color_format);
    const Decoder::ColorFormat decoder_color_format = std::get<0>(ret);
    const unsigned output_planes = std::get<1>(ret);
    const bool keep_original = decoder_keep_original;
    const size_t image_size = max_decoded_width * max_decoded_height * output_planes * sizeof(unsigned char);
    bool skip_decode = _decoder_config._type == DecoderType::SKIP_DECODE;
    // Decode with the height and size equal to a single image
    // File read is done serially since I/O parallelization does not work very well.
    _file_load_time.start();  // Debug timing
    if (_decoder_config._type == DecoderType::SKIP_DECODE) {
        while ((file_counter != _batch_size) && _reader->count_items() > 0) {
            auto read_ptr = buff + image_size * file_counter;
            size_t fsize = _reader->open();
            if (fsize == 0) {
                WRN("Opened file " + _reader->id() + " of size 0");
                continue;
            }

            _actual_read_size[file_counter] = _reader->read_data(read_ptr, fsize);
            if (_actual_read_size[file_counter] < fsize)
                LOG("Reader read less than requested bytes of size: " + _actual_read_size[file_counter]);

            _image_names[file_counter] = _reader->id();
            _reader->close();
            // _compressed_image_size[file_counter] = fsize;
            names[file_counter] = _image_names[file_counter];
            roi_width[file_counter] = max_decoded_width;
            roi_height[file_counter] = max_decoded_height;
            actual_width[file_counter] = max_decoded_width;
            actual_height[file_counter] = max_decoded_height;
            file_counter++;
        }
        //_file_load_time.end();// Debug timing
    } else if (_is_external_source) {
        auto ext_reader = std::static_pointer_cast<ExternalSourceReader>(_reader);
        if (ext_reader->mode() == ExternalSourceFileMode::RAWDATA_UNCOMPRESSED) {
            while ((file_counter != _batch_size) && _reader->count_items() > 0) {
                int width, height, channels;
                unsigned rwidth, rheight;
                auto read_ptr = buff + image_size * file_counter;
                size_t fsize = _reader->open();
                if (fsize == 0) {
                    WRN("Opened file " + _reader->id() + " of size 0");
                    continue;
                }

                _actual_read_size[file_counter] = _reader->read_data(read_ptr, fsize);
                if (_actual_read_size[file_counter] < fsize)
                    LOG("Reader read less than requested bytes of size: " + _actual_read_size[file_counter]);

                _image_names[file_counter] = _reader->id();
                ext_reader->get_dims(file_counter, width, height, channels, rwidth, rheight);
                names[file_counter] = _image_names[file_counter];
                roi_width[file_counter] = rwidth;
                roi_height[file_counter] = rheight;
                actual_width[file_counter] = width;
                actual_height[file_counter] = height;
                _reader->close();
                file_counter++;
            }
            skip_decode = true;
        } else {
            while ((file_counter != _batch_size) && _reader->count_items() > 0) {
                _reader->count_items();
                size_t fsize = _reader->open();
                if (fsize == 0) {
                    WRN("Opened file " + _reader->id() + " of size 0");
                    continue;
                }
                _compressed_buff[file_counter].reserve(fsize);
                _actual_read_size[file_counter] = _reader->read_data(_compressed_buff[file_counter].data(), fsize);
                _image_names[file_counter] = _reader->id();
                _reader->close();
                _compressed_image_size[file_counter] = fsize;
                file_counter++;
            }
        }
        // return LoaderModuleStatus::OK;
    } else {
        while ((file_counter != _batch_size) && _reader->count_items() > 0) {
            size_t fsize = _reader->open();
            if (fsize == 0) {
                WRN("Opened file " + _reader->id() + " of size 0");
                continue;
            }
            _compressed_buff[file_counter].reserve(fsize);
            _actual_read_size[file_counter] = _reader->read_data(_compressed_buff[file_counter].data(), fsize);
            _image_names[file_counter] = _reader->id();
            _reader->close();
            _compressed_image_size[file_counter] = fsize;
            file_counter++;
        }
        if (_randombboxcrop_meta_data_reader) {
            // Fetch the crop co-ordinates for a batch of images
            _bbox_coords = _randombboxcrop_meta_data_reader->get_batch_crop_coords(_image_names);
            set_batch_random_bbox_crop_coords(_bbox_coords);
        } else if (_random_crop_dec_param) {
            _random_crop_dec_param->generate_random_seeds();
        }
    }

    _file_load_time.end();  // Debug timing

    _decode_time.start();  // Debug timing
    if (!skip_decode) {
        for (size_t i = 0; i < _batch_size; i++)
            _decompressed_buff_ptrs[i] = buff + image_size * i;

        const bool is_rocjpeg_decoder = _decoder_config._type == DecoderType::ROCJPEG ||
                                        _decoder_config._type == DecoderType::ROCJPEG_CROPPED;
        if (!is_rocjpeg_decoder) {
#pragma omp parallel for num_threads(_num_threads)
            for (size_t i = 0; i < _batch_size; i++) {
                // initialize the actual decoded height and width with the maximum
                _actual_decoded_width[i] = max_decoded_width;
                _actual_decoded_height[i] = max_decoded_height;
                int original_width, original_height, jpeg_sub_samp;
                if (_decoder[i]->decode_info(_compressed_buff[i].data(), _actual_read_size[i], &original_width, &original_height,
                                            &jpeg_sub_samp) != Decoder::Status::OK) {
                    // Substituting the image which failed decoding with other image from the same batch
                    int j = ((i + 1) != _batch_size) ? _batch_size - 1 : _batch_size - 2;
                    while ((j >= 0)) {
                        if (_decoder[i]->decode_info(_compressed_buff[j].data(), _actual_read_size[j], &original_width, &original_height,
                                                    &jpeg_sub_samp) == Decoder::Status::OK) {
                            _image_names[i] = _image_names[j];
                            _compressed_buff[i] = _compressed_buff[j];
                            _actual_read_size[i] = _actual_read_size[j];
                            _compressed_image_size[i] = _compressed_image_size[j];
                            break;

                        } else
                            j--;
                        if (j < 0) {
                            THROW("All images in the batch failed decoding\n");
                        }
                    }
                }
                _original_height[i] = original_height;
                _original_width[i] = original_width;
                // decode the image and get the actual decoded image width and height
                size_t scaledw, scaledh;
                if (_decoder[i]->is_cropped_decoder()) {
                    if (_randombboxcrop_meta_data_reader) {
                        _decoder[i]->set_bbox_coords(_bbox_coords[i]);
                    } else if (_random_crop_dec_param) {
                        Shape dec_shape = {_original_height[i], _original_width[i]};
                        auto crop_window = _random_crop_dec_param->generate_crop_window(dec_shape, i);
                        _decoder[i]->set_crop_window(crop_window);
                    }
                }
                if (_decoder[i]->decode(_compressed_buff[i].data(), _compressed_image_size[i], _decompressed_buff_ptrs[i],
                                        max_decoded_width, max_decoded_height,
                                        original_width, original_height,
                                        scaledw, scaledh,
                                        decoder_color_format, _decoder_config, keep_original) != Decoder::Status::OK) {
                }
                _actual_decoded_width[i] = scaledw;
                _actual_decoded_height[i] = scaledh;
            }
        } else {
#if ENABLE_HIP
            // Set device ID for load routine thread once
            if (!_set_device_id) {
                hipError_t hip_status = hipSetDevice(_device_id);
                if (hip_status != hipSuccess) {     
                    THROW("hipSetDevice failed");
                }
                _set_device_id = true;
            }
#endif
            std::vector<size_t> rocjpeg_sub_batch_offsets(_rocjpeg_sub_batch_sizes.size(), 0);
            for (size_t shard = 1; shard < _rocjpeg_sub_batch_sizes.size(); shard++) {
                rocjpeg_sub_batch_offsets[shard] = rocjpeg_sub_batch_offsets[shard - 1] + _rocjpeg_sub_batch_sizes[shard - 1];
            }

            const int rocjpeg_decoder_threads = static_cast<int>(_rocjpeg_decoders.size());
            bool rocjpeg_worker_failed = false;
            std::string rocjpeg_worker_error;
            auto record_rocjpeg_worker_error = [&](const std::string& error_message) {
#pragma omp critical(rocjpeg_worker_error)
                {
                    if (!rocjpeg_worker_failed) {
                        rocjpeg_worker_failed = true;
                        rocjpeg_worker_error = error_message;
                    }
                }
            };
#pragma omp parallel for num_threads(rocjpeg_decoder_threads)
            for (size_t shard = 0; shard < _rocjpeg_decoders.size(); shard++) {
#if ENABLE_HIP
                // HIP current device is thread-local; set it for each OpenMP worker.
                hipError_t hip_status = hipSetDevice(_device_id);
                if (hip_status != hipSuccess) {
                    record_rocjpeg_worker_error("hipSetDevice failed inside rocJPEG shard worker");
                    continue;
                }
#endif
                auto& rocjpeg_decoder = _rocjpeg_decoders[shard];
                const size_t shard_begin = rocjpeg_sub_batch_offsets[shard];
                const size_t shard_size = _rocjpeg_sub_batch_sizes[shard];
                const size_t shard_end = shard_begin + shard_size;
                bool shard_failed = false;

                for (size_t i = shard_begin; i < shard_end; i++) {
                    const size_t local_index = i - shard_begin;
                    _actual_decoded_width[i] = max_decoded_width;
                    _actual_decoded_height[i] = max_decoded_height;
                    int original_width, original_height, decoded_width, decoded_height;
                    bool decode_info_found = false;
                    int candidate = static_cast<int>(i);
                    const int shard_begin_index = static_cast<int>(shard_begin);
                    const int shard_end_index = static_cast<int>(shard_end);
                    while (candidate >= shard_begin_index) {
                        if (rocjpeg_decoder->decode_info(_compressed_buff[candidate].data(), _actual_read_size[candidate], &original_width, &original_height,
                                                         &decoded_width, &decoded_height,
                                                         max_decoded_width, max_decoded_height, decoder_color_format, static_cast<int>(local_index)) == Decoder::Status::OK) {
                            if (candidate != static_cast<int>(i)) {
                                _image_names[i] = _image_names[candidate];
                                _compressed_buff[i] = _compressed_buff[candidate];
                                _actual_read_size[i] = _actual_read_size[candidate];
                                _compressed_image_size[i] = _compressed_image_size[candidate];
                            }
                            decode_info_found = true;
                            break;
                        }

                        if (candidate == static_cast<int>(i)) {
                            candidate = shard_end_index - 1;
                        } else {
                            candidate--;
                        }
                        if (candidate == static_cast<int>(i)) {
                            candidate--;
                        }
                    }
                    if (!decode_info_found) {
                        record_rocjpeg_worker_error("All images in the rocJpeg sub-batch failed decoding\n");
                        shard_failed = true;
                        break;
                    }
                    _original_height[i] = original_height;
                    _original_width[i] = original_width;
                    _actual_decoded_width[i] = decoded_width;
                    _actual_decoded_height[i] = decoded_height;

                    if (rocjpeg_decoder->is_cropped_decoder()) {
                        if (_randombboxcrop_meta_data_reader) {
                            rocjpeg_decoder->set_bbox_coords(_bbox_coords[i]);
                        } else if (_random_crop_dec_param) {
                            Shape dec_shape = {_original_height[i], _original_width[i]};
                            auto crop_window = _random_crop_dec_param->generate_crop_window(dec_shape, i);
                            rocjpeg_decoder->set_crop_window(crop_window);
                        }
                    }
                }
                if (shard_failed) {
                    continue;
                }

                std::vector<unsigned char *> shard_output(_decompressed_buff_ptrs.begin() + shard_begin, _decompressed_buff_ptrs.begin() + shard_end);
                std::vector<size_t> shard_original_width(_original_width.begin() + shard_begin, _original_width.begin() + shard_end);
                std::vector<size_t> shard_original_height(_original_height.begin() + shard_begin, _original_height.begin() + shard_end);
                std::vector<size_t> shard_actual_decoded_width(_actual_decoded_width.begin() + shard_begin, _actual_decoded_width.begin() + shard_end);
                std::vector<size_t> shard_actual_decoded_height(_actual_decoded_height.begin() + shard_begin, _actual_decoded_height.begin() + shard_end);

                if (rocjpeg_decoder->decode_batch(shard_output,
                                                  max_decoded_width, max_decoded_height,
                                                  shard_original_width, shard_original_height,
                                                  shard_actual_decoded_width, shard_actual_decoded_height) != Decoder::Status::OK) {
                    record_rocjpeg_worker_error("rocJpeg sub-batch decode failed\n");
                    continue;
                }

                std::copy(shard_actual_decoded_width.begin(), shard_actual_decoded_width.end(), _actual_decoded_width.begin() + shard_begin);
                std::copy(shard_actual_decoded_height.begin(), shard_actual_decoded_height.end(), _actual_decoded_height.begin() + shard_begin);
            }
            if (rocjpeg_worker_failed) {
                THROW(rocjpeg_worker_error);
            }
        }

        for (size_t i = 0; i < _batch_size; i++) {
            names[i] = _image_names[i];
            roi_width[i] = _actual_decoded_width[i];
            roi_height[i] = _actual_decoded_height[i];
            actual_width[i] = _original_width[i];
            actual_height[i] = _original_height[i];
        }
    }
    _bbox_coords.clear();
    _decode_time.end();  // Debug timing
    return LoaderModuleStatus::OK;
}
