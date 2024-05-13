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

#include "decoders/video/rocdecode_video_decoder.h"

#include "pipeline/commons.h"
#include <stdio.h>

#ifdef ROCAL_VIDEO && ENABLE_HIP
RocDecodeVideoDecoder::RocDecodeVideoDecoder(){};

int RocDecodeVideoDecoder::seek_frame(AVRational avg_frame_rate, AVRational time_base, unsigned frame_number) {

}

// int RocDecodeVideoDecoder::hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type, AVBufferRef *hw_device_ctx) {

// }

// Seeks to the frame_number in the video file and decodes each frame in the sequence.
VideoDecoder::Status RocDecodeVideoDecoder::Decode(unsigned char *out_buffer, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, AVPixelFormat out_pix_format) {
    
    int n_video_bytes = 0, n_frames_returned = 0, n_frame = 0;
    uint8_t *p_video = nullptr;
    uint8_t *p_frame = nullptr;
    int64_t pts = 0;
    OutputSurfaceInfo *surf_info;
    OutputSurfaceInfo *resize_surf_info = nullptr;
    Dim resize_dim = {};
    size_t rgb_image_size, resize_image_size;
    OutputFormatEnum e_output_format = rgb;
    int frames = 0;
    bool sequence_decoded = true;
    uint8_t* frame_buffers;
    do {
        // auto start_time = std::chrono::high_resolution_clock::now();
        // if (seek_criteria == 1 && first_frame) {
        //     // use VideoSeekContext class to seek to given frame number
        //     video_seek_ctx.seek_frame_ = seek_to_frame;
        //     video_seek_ctx.seek_crit_ = SEEK_CRITERIA_FRAME_NUM;
        //     video_seek_ctx.seek_mode_ = (seek_mode ? SEEK_MODE_EXACT_FRAME : SEEK_MODE_PREV_KEY_FRAME);
        //     _demuxer->Seek(video_seek_ctx, &p_video, &n_video_bytes);
        //     pts = video_seek_ctx.out_frame_pts_;
        //     std::cout << "info: Number of frames that were decoded during seek - " << video_seek_ctx.num_frames_decoded_ << std::endl;
        //     first_frame = false;
        // } else if (seek_criteria == 2 && first_frame) {
        //     // use VideoSeekContext class to seek to given timestamp
        //     video_seek_ctx.seek_frame_ = seek_to_frame;
        //     video_seek_ctx.seek_crit_ = SEEK_CRITERIA_TIME_STAMP;
        //     video_seek_ctx.seek_mode_ = (seek_mode ? SEEK_MODE_EXACT_FRAME : SEEK_MODE_PREV_KEY_FRAME);
        //     _demuxer->Seek(video_seek_ctx, &p_video, &n_video_bytes);
        //     pts = video_seek_ctx.out_frame_pts_;
        //     std::cout << "info: Duration of frame found after seek - " << video_seek_ctx.out_frame_duration_ << " ms" << std::endl;
        //     first_frame = false;
        // } else {
        //     _demuxer->Demux(&p_video, &n_video_bytes, &pts);
        // }
        // std::cerr << "Decoder pts :: " << pts << "\n";


        _seek_ctx.seek_frame_ = 0;
        _seek_ctx.seek_crit_ = SEEK_CRITERIA_FRAME_NUM;
        _seek_ctx.seek_mode_ = SEEK_MODE_PREV_KEY_FRAME;
        _demuxer->Seek(_seek_ctx, &p_video, &n_video_bytes);
        // _demuxer->Demux(&p_video, &n_video_bytes, &pts);

        n_frames_returned = _decoder->DecodeFrame(p_video, n_video_bytes, 0, pts);
        if (!n_frame && !_decoder->GetOutputSurfaceInfo(&surf_info)) {
            std::cerr << "Error: Failed to get Output Image Info!" << std::endl;
            break;
        }
        if (resize_dim.w && resize_dim.h && !resize_surf_info) {    // TO be fixed
            resize_surf_info = new OutputSurfaceInfo;
            memcpy(resize_surf_info, surf_info, sizeof(OutputSurfaceInfo));
        }

        int last_index = 0;
        for (int i = 0; i < n_frames_returned; i++, frames++) {
            if (frames == sequence_length) {
                sequence_decoded = true;
            }
            p_frame = _decoder->GetFrame(&pts);
            std::cerr << "Output surface info : " << surf_info->output_surface_size_in_bytes << "\n";
            // if (frame_buffers == nullptr) {
            //     // for (int i = 0; i < frame_buffers_size; i++) {
            //         HIP_API_CALL(hipMalloc(&frame_buffers, surf_info->output_surface_size_in_bytes));
            //     // }
            // }
            // copy the decoded frame into the frame_buffers at current_frame_index
            // HIP_API_CALL(hipMemcpyDtoDAsync(frame_buffers, p_frame, surf_info->output_surface_size_in_bytes, _decoder->GetStream()));
            
            // allocate extra device memories to use double-buffering for keeping two decoded frames
            // if (frame_buffers[0] == nullptr) {
            //     for (int i = 0; i < frame_buffers_size; i++) {
            //         HIP_API_CALL(hipMalloc(&frame_buffers[i], surf_info->output_surface_size_in_bytes));
            //     }
            // }
            std::cerr << "OUT W : " << surf_info->output_width << "\n";
            std::cerr << "OUT H : " << surf_info->output_height << "\n";

            int rgb_width;
            if (surf_info->bit_depth == 8) {
                rgb_width = (surf_info->output_width + 1) & ~1; // has to be a multiple of 2 for hip colorconvert kernels
                rgb_image_size = ((e_output_format == bgr) || (e_output_format == rgb)) ? rgb_width * surf_info->output_height * 3 : rgb_width * surf_info->output_height * 4;
            } else {
                rgb_width = (surf_info->output_width + 1) & ~1;
                rgb_image_size = ((e_output_format == bgr) || (e_output_format == rgb)) ? rgb_width * surf_info->output_height * 3 : ((e_output_format == bgr48) || (e_output_format == rgb48)) ? 
                                                        rgb_width * surf_info->output_height * 6 : rgb_width * surf_info->output_height * 8;
            }
            // if (p_rgb_dev_mem == nullptr) {
            //     hip_status = hipMalloc(&p_rgb_dev_mem, rgb_image_size);
            //     if (hip_status != hipSuccess) {
            //         std::cerr << "ERROR: hipMalloc failed to allocate the device memory for the output!" << hip_status << std::endl;
            //         return;
            //     }
            // }
            _post_process.ColorConvertYUV2RGB(p_frame, surf_info, out_buffer, e_output_format, _decoder->GetStream());

            _decoder->ReleaseFrame(pts);
            std::cerr << "Total size : " << out_stride * out_width << "\n";
            out_buffer += (out_height * 3 * out_width);
        }
        n_frame += n_frames_returned;
        if (sequence_decoded) break;
    } while (n_video_bytes);

    // if (frame_buffers) {
    //     auto hip_status = hipFree(frame_buffers);
    //     frame_buffers = nullptr;
    //     if (hip_status != hipSuccess) {
    //         std::cout << "ERROR: hipFree failed! (" << hip_status << ")" << std::endl;
    //     }   
    // }

    std::cerr << "Total number of frames decoded : " << n_frame << "\n";
    return VideoDecoder::Status::OK;
}

// Initialize will open a new decoder and initialize the context
VideoDecoder::Status RocDecodeVideoDecoder::Initialize(const char *src_filename) {
    _src_filename = src_filename;
    _demuxer = std::make_unique<VideoDemuxer>(src_filename);
    _rocdec_codec_id = AVCodec2RocDecVideoCodec(_demuxer->GetCodecID());
    std::cerr << "CODEC ID : " << (int)_rocdec_codec_id << "\n";
    _decoder = std::make_unique<RocVideoDecoder>(0, _surface_mem_type, _rocdec_codec_id);  // Exact device ID to be set here
    _stream = _decoder->GetStream();
    return VideoDecoder::Status::OK;

}

void RocDecodeVideoDecoder::release() {
}

RocDecodeVideoDecoder::~RocDecodeVideoDecoder() {
}
#endif
