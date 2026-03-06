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

#pragma once
#include "decoders/image/decoder.h"

#if ENABLE_ROCJPEG

#include "rocjpeg/rocjpeg.h"

/**
 * @brief Gets the channel pitch and sizes.
 *
 * This function gets the channel pitch and sizes based on the specified output format, chroma subsampling,
 * output image, and channel sizes.
 *
 * @param output_format The output format.
 * @param subsampling The chroma subsampling.
 * @param widths The array to store the channel widths.
 * @param heights The array to store the channel heights.
 * @param num_channels The number of channels.
 * @param output_image The output image.
 * @param channel_sizes The array to store the channel sizes.
 * @return The channel pitch.
 */
inline int GetChannelPitchAndSizes(RocJpegDecodeParams decode_params, RocJpegChromaSubsampling subsampling, uint32_t *widths, uint32_t *heights,
                            uint32_t &num_channels, RocJpegImage &output_image, uint32_t *channel_sizes) {
    uint32_t roi_width = decode_params.crop_rectangle.right - decode_params.crop_rectangle.left;
    uint32_t roi_height = decode_params.crop_rectangle.bottom - decode_params.crop_rectangle.top;
    if (roi_width > widths[0] || roi_height > heights[0]) {
        ERR("Invalid ROI passed to the decoder")
    }

    switch (decode_params.output_format) {
        case ROCJPEG_OUTPUT_NATIVE:
            switch (subsampling) {
                case ROCJPEG_CSS_444:
                    num_channels = 3;
                    output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = widths[0];
                    channel_sizes[2] = channel_sizes[1] = channel_sizes[0] = output_image.pitch[0] * heights[0];
                    break;
                case ROCJPEG_CSS_440:
                    num_channels = 3;
                    output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = widths[0];
                    channel_sizes[0] = output_image.pitch[0] * heights[0];
                    channel_sizes[2] = channel_sizes[1] = output_image.pitch[0] * (heights[0] >> 1);
                    break;
                case ROCJPEG_CSS_422:
                    num_channels = 1;
                    output_image.pitch[0] = widths[0] * 2;
                    channel_sizes[0] = output_image.pitch[0] * heights[0];
                    break;
                case ROCJPEG_CSS_420:
                    num_channels = 2;
                    output_image.pitch[1] = output_image.pitch[0] = widths[0];
                    channel_sizes[0] = output_image.pitch[0] * heights[0];
                    channel_sizes[1] = output_image.pitch[1] * (heights[0] >> 1);
                    break;
                case ROCJPEG_CSS_400:
                    num_channels = 1;
                    output_image.pitch[0] = widths[0];
                    channel_sizes[0] = output_image.pitch[0] * heights[0];
                    break;
                default:
                    std::cout << "Unknown chroma subsampling!" << std::endl;
                    return EXIT_FAILURE;
            }
            break;
        case ROCJPEG_OUTPUT_YUV_PLANAR:
            if (subsampling == ROCJPEG_CSS_400) {
                num_channels = 1;
                output_image.pitch[0] = widths[0];
                channel_sizes[0] = output_image.pitch[0] * heights[0];
            } else {
                num_channels = 3;
                output_image.pitch[0] = widths[0];
                output_image.pitch[1] = widths[1];
                output_image.pitch[2] = widths[2];
                channel_sizes[0] = output_image.pitch[0] * heights[0];
                channel_sizes[1] = output_image.pitch[1] * heights[1];
                channel_sizes[2] = output_image.pitch[2] * heights[2];
            }
            break;
        case ROCJPEG_OUTPUT_Y:
            num_channels = 1;
            output_image.pitch[0] = widths[0];
            channel_sizes[0] = output_image.pitch[0] * heights[0];
            break;
        case ROCJPEG_OUTPUT_RGB:
            num_channels = 1;
            output_image.pitch[0] = (widths[0]) * 3;
            channel_sizes[0] = output_image.pitch[0] * heights[0];
            break;
        case ROCJPEG_OUTPUT_RGB_PLANAR:
            num_channels = 3;
            output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = widths[0];
            channel_sizes[2] = channel_sizes[1] = channel_sizes[0] = output_image.pitch[0] * heights[0];
            break;
        default:
            std::cout << "Unknown output format!" << std::endl;
            return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

/**
 * @brief Gets the chroma subsampling string.
 *
 * This function gets the chroma subsampling string based on the specified subsampling value.
 *
 * @param subsampling The chroma subsampling value.
 * @param chroma_sub_sampling The string to store the chroma subsampling.
 */
inline void GetChromaSubsamplingStr(RocJpegChromaSubsampling subsampling, std::string &chroma_sub_sampling) {
    switch (subsampling) {
        case ROCJPEG_CSS_444:
            chroma_sub_sampling = "YUV 4:4:4";
            break;
        case ROCJPEG_CSS_440:
            chroma_sub_sampling = "YUV 4:4:0";
            break;
        case ROCJPEG_CSS_422:
            chroma_sub_sampling = "YUV 4:2:2";
            break;
        case ROCJPEG_CSS_420:
            chroma_sub_sampling = "YUV 4:2:0";
            break;
        case ROCJPEG_CSS_411:
            chroma_sub_sampling = "YUV 4:1:1";
            break;
        case ROCJPEG_CSS_400:
            chroma_sub_sampling = "YUV 4:0:0";
            break;
        case ROCJPEG_CSS_UNKNOWN:
            chroma_sub_sampling = "UNKNOWN";
            break;
        default:
            chroma_sub_sampling = "";
            break;
    }
}

class HWRocJpegDecoder : public Decoder {
   public:
    //! Default constructor
    HWRocJpegDecoder();
    HWRocJpegDecoder(hipStream_t &stream) { _hip_stream = stream; }
    //! Decodes the header of the Jpeg compressed data and returns basic info about the compressed image
    /*!
     \param input_buffer  User provided buffer containig the encoded image
     \param input_size Size of the compressed data provided in the input_buffer
     \param width pointer to the user's buffer to write the width of the compressed image to
     \param height pointer to the user's buffer to write the height of the compressed image to
     \param color_comps pointer to the user's buffer to write the number of color components of the compressed image to
    */
    Status decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *color_comps) override;
    
    //! Decodes the header of the Jpeg compressed data and returns basic info about the compressed image
    //! It also scales the width and height wrt max decoded width and height
    /*!
     \param input_buffer  User provided buffer containig the encoded image
     \param input_size Size of the compressed data provided in the input_buffer
     \param width pointer to the user's buffer to write the width of the compressed image/scaled width based on max_decoded_width
     \param height pointer to the user's buffer to write the height of the compressed image/scaled height based on max_decoded_height
     \param actual_width pointer to the user's buffer to write the width of the compressed image
     \param actual_height pointer to the user's buffer to write the height of the compressed image
     \param max_decoded_width maximum width of the decoded image
     \param max_decoded_height maximum height of the decoded image
     \param desired_decoded_color_format user provided color format of the decoded image
     \param index index of the image in the batch for which decode info must be fetched
    */
    Status decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *actual_width, int *actual_height, int max_decoded_width, int max_decoded_height, Decoder::ColorFormat desired_decoded_color_format, int index) override;

    Decoder::Status decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                           size_t max_decoded_width, size_t max_decoded_height,
                           size_t original_image_width, size_t original_image_height,
                           size_t &actual_decoded_width, size_t &actual_decoded_height,
                           Decoder::ColorFormat desired_decoded_color_format, DecoderConfig config, bool keep_original_size = false) override { return Status::UNSUPPORTED; }


    //! Decodes a batch of actual image data
    /*!
      \param output_buffer User provided HIP buffer used to write the decoded image into
      \param max_decoded_width The maximum width user wants the decoded image to be. Image will be downscaled if bigger.
      \param max_decoded_height The maximum height user wants the decoded image to be. Image will be downscaled if bigger.
      \param original_image_width The actual width of the compressed image. decoded width will be equal to this if this is smaller than max_decoded_width
      \param original_image_height The actual height of the compressed image. decoded height will be equal to this if this is smaller than max_decoded_height
      \param actual_decoded_width The width of the image after decoding and scaling if original width is greater than max decoded width
      \param actual_decoded_height The height of the image after decoding and scaling if original height is greater than max decoded height
    */
    Decoder::Status decode_batch(std::vector<unsigned char *> &output_buffer,
                                 size_t max_decoded_width, size_t max_decoded_height,
                                 std::vector<size_t> original_image_width, std::vector<size_t> original_image_height,
                                 std::vector<size_t> &actual_decoded_width, std::vector<size_t> &actual_decoded_height) override;

    ~HWRocJpegDecoder() override;
    void initialize(int device_id) override {}
    void initialize(int device_id, unsigned batch_size) override;
    bool is_cropped_decoder() override { return false; }
    void set_bbox_coords(std::vector<float> bbox_coord) override { _bbox_coord = bbox_coord; }
    std::vector<float> get_bbox_coords() override { return _bbox_coord; }
    void set_crop_window(CropWindow &crop_window) override { _crop_window = crop_window; }
   private:
    std::vector<float> _bbox_coord;
    CropWindow _crop_window;
    RocJpegHandle _rocjpeg_handle = nullptr;
    std::vector<RocJpegStreamHandle> _rocjpeg_streams;
    unsigned _batch_size;
    void * _rocjpeg_image_buff = nullptr;
    size_t _rocjpeg_image_buff_size = 0, _prev_image_buff_size = 0;
    size_t *_src_width = nullptr;
    size_t *_src_height = nullptr;
    size_t *_dst_width = nullptr, *_dst_height = nullptr;
    size_t *_src_hstride = nullptr, *_src_img_offset = nullptr;
    uint32_t *_dst_img_idx = nullptr;  // Maps resize-subset index -> original batch index in output tensor.
    std::vector<bool> _image_needs_rescaling;   // A flag for each image in the batch, set to `true` if the image needs rescaling.
    std::vector<RocJpegImage> _output_images = {};
    std::vector<RocJpegDecodeParams> _decode_params = {};
    uint32_t _num_channels = 0;
    bool _enable_resize = false;
    int _device_id = 0;
    hipStream_t _hip_stream = nullptr;
};
#endif
