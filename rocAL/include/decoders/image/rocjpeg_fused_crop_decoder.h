/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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
class FusedCropRocJpegDecoder : public Decoder {
   public:
    //! Default constructor
    FusedCropRocJpegDecoder() {}
    //! Decodes the header of the jpeg compressed data and returns basic info about the compressed image
    /*!
     \param input_buffer  User provided buffer containing the encoded image
     \param input_size Size of the compressed data provided in the input_buffer
     \param width pointer to the user's buffer to write the width of the compressed image
     \param height pointer to the user's buffer to write the height of the compressed image
     \param color_comps pointer to the user's buffer to write the number of color components of the compressed image to
    */
    Status decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *color_comps) override;

    //! Decodes the header of the jpeg compressed data and returns basic info about the compressed image
    //! It also scales the width and height wrt max decoded width and height
    /*!
     \param input_buffer  User provided buffer containing the encoded image
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

    ~FusedCropRocJpegDecoder() override;
    void initialize(int device_id) override {}
    void initialize(int device_id, unsigned batch_size) override;
    bool is_cropped_decoder() override { return true; }
    void set_bbox_coords(std::vector<float> bbox_coord) override;
    std::vector<float> get_bbox_coords() override { return _bbox_coord; }
    void set_crop_window(CropWindow &crop_window) override;

   private:
    std::vector<float> _bbox_coord;
    CropWindow _crop_window;
    RocJpegHandle _rocjpeg_handle;
    std::vector<RocJpegStreamHandle> _rocjpeg_streams;
    std::vector<RocJpegImage> _output_images = {};
    std::vector<RocJpegDecodeParams> _decode_params_batch;
    RocJpegDecodeParams *_decode_params;
    unsigned _batch_size;
    int _max_decoded_width, _max_decoded_height, _original_image_width, _original_image_height;
};
#endif
