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

#include "decoders/video/rocdec_video_decoder.h"

#include "pipeline/commons.h"
#include <stdio.h>

#ifdef ROCAL_VIDEO
#ifdef ENABLE_HIP

RocDecVideoDecoder::RocDecVideoDecoder(){};

int RocDecVideoDecoder::seek_frame(unsigned frame_number, uint8_t **video, int* video_bytes) {
    _seek_ctx.seek_frame_ = frame_number;
    _demuxer->Seek(_seek_ctx, video, video_bytes);
    int64_t dts;
    auto pts = _seek_ctx.out_frame_pts_;
    // if (_seek_ctx.req_frame_dts_ != _seek_ctx.out_frame_dts_) {
    //     // Seek upto requested frame is received
    //     auto n_frames_returned = _decoder->DecodeFrame(*video, *video_bytes, 0, pts);
    //     do {
    //         _demuxer->Demux(video, video_bytes, &pts, &dts);
    //         auto n_frames_returned = _decoder->DecodeFrame(*video, *video_bytes, 0, pts);
    //     } while (dts < _seek_ctx.req_frame_dts_);
    // }
    // std::cerr << "Frame Number : " << frame_number << "\t";
    // std::cerr << "Requested Frame dts : " << _seek_ctx.req_frame_dts_ << "\t";
    // std::cerr << "Out Frame dts : " << _seek_ctx.out_frame_dts_ << "\t";
    // std::cerr << "DTS at end :: " << dts << "\n";
    return pts;
    // Add support to seek to the exact frame
}

// int RocDecVideoDecoder::hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type, AVBufferRef *hw_device_ctx) {

// }

// Seeks to the frame_number in the video file and decodes each frame in the sequence.
VideoDecoder::Status RocDecVideoDecoder::Decode(unsigned char *out_buffer, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, AVPixelFormat out_pix_format) {
    
    int n_video_bytes = 0, n_frames_returned = 0, n_frame = 0;
    uint8_t *p_video = nullptr;
    uint8_t *p_frame = nullptr;
    int64_t pts = 0;
    OutputSurfaceInfo *surf_info;
    OutputSurfaceInfo *resize_surf_info = nullptr;
    Dim resize_dim = {};
    size_t rgb_image_size, resize_image_size;
    OutputFormatEnum e_output_format = rgb;
    int frames = 1;
    bool sequence_decoded = false;
    uint8_t* frame_buffers;
    bool seek_first_frame = true;
    do {
        if (seek_first_frame) {
            pts = seek_frame(seek_frame_number, &p_video, &n_video_bytes);    // Seek the first frame in the sequence
            seek_first_frame = false;
        } else {
            _demuxer->Demux(&p_video, &n_video_bytes, &pts);
        }
        n_frames_returned = _decoder->DecodeFrame(p_video, n_video_bytes, 0, pts);
        
        // If no frames are returned Demux again, untill the decoder returns valid frames
        if (!n_frames_returned) {
            // seek_first_frame = true;
            continue;
        }
        if (!n_frame && !_decoder->GetOutputSurfaceInfo(&surf_info)) {
            std::cerr << "Error: Failed to get Output Image Info!" << std::endl;
            break;
        }
        // Allocate frame buffers
        if (_frame_buffers.size() == 0) {
            _frame_buffers.resize(sequence_length);
            for (int i = 0; i < sequence_length; i++) {
                HIP_API_CALL(hipMalloc(&_frame_buffers[i], surf_info->output_surface_size_in_bytes));
            }
        }
        // Resize portion TBA
        if (resize_dim.w && resize_dim.h && !resize_surf_info) { 
            resize_surf_info = new OutputSurfaceInfo;
            memcpy(resize_surf_info, surf_info, sizeof(OutputSurfaceInfo));
        }
        
        // Take the min of sequence length and num frames to avoid out of bounds memory error
        int required_n_frames = std::min(static_cast<int>(sequence_length), n_frames_returned);
        for (int i = 0; i < required_n_frames; i++, frames++) {
            std::cerr << "Frame : " << frames << "\n";
            if (frames == sequence_length) {
                sequence_decoded = true;
            }
            p_frame = _decoder->GetFrame(&pts);

            HIP_API_CALL(hipMemcpyDtoD(_frame_buffers[frames - 1], p_frame, surf_info->output_surface_size_in_bytes));

            // TODO - This portion can be moved outside the loop ?
            int rgb_width;
            if (surf_info->bit_depth == 8) {
                rgb_width = (surf_info->output_width + 1) & ~1; // has to be a multiple of 2 for hip colorconvert kernels
                rgb_image_size = ((e_output_format == bgr) || (e_output_format == rgb)) ? rgb_width * surf_info->output_height * 3 : rgb_width * surf_info->output_height * 4;
            } else {
                rgb_width = (surf_info->output_width + 1) & ~1;
                rgb_image_size = ((e_output_format == bgr) || (e_output_format == rgb)) ? rgb_width * surf_info->output_height * 3 : ((e_output_format == bgr48) || (e_output_format == rgb48)) ? 
                                                        rgb_width * surf_info->output_height * 6 : rgb_width * surf_info->output_height * 8;
            }

            _post_process.ColorConvertYUV2RGB(_frame_buffers[frames - 1], surf_info, out_buffer, e_output_format, _decoder->GetStream());

            _decoder->ReleaseFrame(pts);
            out_buffer += (out_height * 3 * out_width);
        }
        n_frame += required_n_frames;
        if (sequence_decoded) {
            std::cerr << "Sequence decoded : " << n_frame << "\n";
            // sync to finish copy
            if (hipStreamSynchronize(_decoder->GetStream()) != hipSuccess)
                THROW("hipStreamSynchronize failed for hipMemcpy ")
            break;
        }
    } while (n_video_bytes);

    std::cerr << "Total number of frames decoded : " << n_frame << "\n";
    return VideoDecoder::Status::OK;
}

// Initialize will open a new decoder and initialize the context
VideoDecoder::Status RocDecVideoDecoder::Initialize(const char *src_filename) {
    _src_filename = src_filename;
    _demuxer = std::make_unique<VideoDemuxer>(src_filename);
    _rocdec_codec_id = AVCodec2RocDecVideoCodec(_demuxer->GetCodecID());
    std::cerr << "CODEC ID : " << (int)_rocdec_codec_id << "\n";
    _decoder = std::make_unique<RocVideoDecoder>(0, _surface_mem_type, _rocdec_codec_id);  // Exact device ID to be set here
    _stream = _decoder->GetStream();

    // Seek context Init
    _seek_ctx.seek_frame_ = 0;
    _seek_ctx.seek_crit_ = SEEK_CRITERIA_FRAME_NUM;
    _seek_ctx.seek_mode_ = SEEK_MODE_PREV_KEY_FRAME;
    return VideoDecoder::Status::OK;

}

void RocDecVideoDecoder::release() {
    if (_frame_buffers.size() != 0) {
        for (int i = 0; i < _frame_buffers.size(); i++) {
            if (_frame_buffers[i])
                HIP_API_CALL(hipFree(_frame_buffers[i]));
        }
    }
}

RocDecVideoDecoder::~RocDecVideoDecoder() {
}
#endif
#endif
