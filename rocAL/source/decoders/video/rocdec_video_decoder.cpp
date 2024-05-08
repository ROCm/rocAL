/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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
#include <stdio.h>
#include "pipeline/commons.h"
#include "decoders/video/rocdec_video_decoder.h"

#ifdef ROCAL_VIDEO
RocDecVideoDecoder::RocDecVideoDecoder(){};

// Initialize will open a new decoder and initialize the context
VideoDecoder::Status RocDecVideoDecoder::Initialize(const char *src_filename, int device_id) {

    VideoDecoder::Status status = Status::OK;
    
    // create rocDecoder and Demuxer for rocDecode
    OutputSurfaceMemoryType mem_type = OUT_SURFACE_MEM_DEV_INTERNAL;      // set to internal
    _demuxer = std::make_shared<VideoDemuxer>(src_filename);
    rocDecVideoCodec rocdec_codec_id = AVCodec2RocDecVideoCodec(_demuxer->GetCodecID());
    _rocvid_decoder = std::make_shared<RocVideoDecoder>(device_id, mem_type, rocdec_codec_id, 0, nullptr, 0);

    std::string device_name, gcn_arch_name;
    int pci_bus_id, pci_domain_id, pci_device_id;
    _rocvid_decoder->GetDeviceinfo(device_name, gcn_arch_name, pci_bus_id, pci_domain_id, pci_device_id);
    std::cout << "info: Using GPU device " << device_id << " - " << device_name << "[" << gcn_arch_name << "] on PCI bus " <<
    std::setfill('0') << std::setw(2) << std::right << std::hex << pci_bus_id << ":" << std::setfill('0') << std::setw(2) <<
    std::right << std::hex << pci_domain_id << "." << pci_device_id << std::dec << std::endl;

    return status;
}

// Seeks to the frame_number in the video file and decodes each frame in the sequence.
VideoDecoder::Status RocDecVideoDecoder::Decode(unsigned char *out_buffer, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, AVPixelFormat out_pix_format) {
    
    VideoDecoder::Status status = Status::OK;
    VideoSeekContext video_seek_ctx;
    if (!_demuxer || !_rocvid_decoder || !out_buffer) {
        ERR("RocDecVideoDecoder::Decoder is not initialized");
        return Status::FAILED;        
    }
    if (!out_buffer || !(sequence_length|stride)) {
        ERR("RocDecVideoDecoder::Invalid parameter passed");
        return Status::FAILED;        
    }

    int64_t pts = 0;
    int n_video_bytes = 0, n_frame_returned = 0, n_frame = 0, pkg_flags = 0;
    OutputSurfaceInfo *surf_info;
    uint8_t *pvideo = nullptr;
    int num_decoded_frames = sequence_length * stride;
    bool b_seek = !seek_frame_number;       // only if not first frame 
    uint32_t image_size = out_height * out_stride * sizeof(uint8_t);
    video_seek_ctx.seek_crit_ = SEEK_CRITERIA_FRAME_NUM;
    video_seek_ctx.seek_mode_ = SEEK_MODE_PREV_KEY_FRAME;
    do {
        if (b_seek) {
            video_seek_ctx.seek_frame_ = static_cast<uint64_t>(seek_frame_number);
            video_seek_ctx.seek_crit_ = SEEK_CRITERIA_FRAME_NUM;
            video_seek_ctx.seek_mode_ = SEEK_MODE_PREV_KEY_FRAME;
            _demuxer->Seek(video_seek_ctx, &pvideo, &n_video_bytes);
            pts = video_seek_ctx.out_frame_pts_;
            std::cout << "info: Number of frames that were decoded during seek - " << video_seek_ctx.num_frames_decoded_ << std::endl;
            b_seek = false;
        } else {
            _demuxer->Demux(&pvideo, &n_video_bytes, &pts);
        }
        // Treat 0 bitstream size as end of stream indicator
        if (n_video_bytes == 0) {
            pkg_flags |= ROCDEC_PKT_ENDOFSTREAM;
        }
        n_frame_returned = _rocvid_decoder->DecodeFrame(pvideo, n_video_bytes, pkg_flags, pts);
        if (!n_frame && !_rocvid_decoder->GetOutputSurfaceInfo(&surf_info)) {
            std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
            break;
        }
        for (int i = 0; i < n_frame_returned; i++) {
            uint8_t *pframe = _rocvid_decoder->GetFrame(&pts);
            if (n_frame % stride == 0) {
                // todo:: access the frame in HIP/GPU memory
                // do color conversion and scaling following videodecodergb sample
                // the destination memory should be set to out_buffer pointer (expect to be GPU mem ptr)
                out_buffer += image_size;
            }
            // release frame
            _rocvid_decoder->ReleaseFrame(pts);
        }
        //auto end_time = std::chrono::high_resolution_clock::now();
        //auto time_per_decode = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        //total_dec_time += time_per_decode;
        n_frame += n_frame_returned;
        if (n_frame >= num_decoded_frames) {
            break;
        }
    } while (n_video_bytes);

    return status;
}


void RocDecVideoDecoder::release() {
    // todo:: release resources allocated for this decode instance
}

RocDecVideoDecoder::~RocDecVideoDecoder() {
    release();
}
#endif
