/*
Copyright (c) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include <iostream>
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #if USE_AVCODEC_GREATER_THAN_58_134
        #include <libavcodec/bsf.h>
    #endif
}

#include "rocdecode.h"
/*!
 * \file
 * \brief The AMD Video Demuxer for rocDecode Library.
 *
 * \defgroup group_amd_rocdecode_videodemuxer videoDemuxer: AMD rocDecode Video Demuxer API
 * \brief AMD The rocDecode video demuxer API.
 */

/**
 * @brief Enum for Seek mode
 * 
 */
typedef enum SeekModeEnum {
    SEEK_MODE_EXACT_FRAME = 0,
    SEEK_MODE_PREV_KEY_FRAME = 1,
    SEEK_MODE_NUM,
} SeekMode;

/**
 * @brief Enum for Seek Criteria
 * 
 */
typedef enum SeekCriteriaEnum {
    SEEK_CRITERIA_FRAME_NUM = 0,
    SEEK_CRITERIA_TIME_STAMP = 1,
    SEEK_CRITERIA_NUM,
} SeekCriteria;

struct PacketData {
    int32_t key;
    int64_t pts;
    int64_t dts;
    uint64_t pos;
    uintptr_t bsl_data;
    uint64_t bsl;
    uint64_t duration;
};

class VideoSeekContext {
public:
    VideoSeekContext()
        : use_seek_(false), seek_frame_(0), seek_mode_(SEEK_MODE_PREV_KEY_FRAME), seek_crit_(SEEK_CRITERIA_FRAME_NUM),
        out_frame_pts_(0), out_frame_duration_(0), num_frames_decoded_(0U) {}

    VideoSeekContext(uint64_t frame_id)
        : use_seek_(true), seek_frame_(frame_id), seek_mode_(SEEK_MODE_PREV_KEY_FRAME),
        seek_crit_(SEEK_CRITERIA_FRAME_NUM), out_frame_pts_(0), out_frame_duration_(0), num_frames_decoded_(0U) {}

    VideoSeekContext& operator=(const VideoSeekContext& other) {
        use_seek_ = other.use_seek_;
        seek_frame_ = other.seek_frame_;
        seek_mode_ = other.seek_mode_;
        seek_crit_ = other.seek_crit_;
        out_frame_pts_ = other.out_frame_pts_;
        out_frame_duration_ = other.out_frame_duration_;
        num_frames_decoded_ = other.num_frames_decoded_;
        return *this;
    }

    /* Will be set to false when not seeking, true otherwise;
     */
    bool use_seek_;

    /* Frame we want to get. Set by user.
     * Shall be set to frame timestamp in case seek is done by time.
     */
    uint64_t seek_frame_;

    /* Mode in which we seek. */
    SeekMode seek_mode_;

    /* Criteria by which we seek. */
    SeekCriteria seek_crit_;

    /* PTS of frame found after seek. */
    int64_t out_frame_pts_;

    /* Duration of frame found after seek. */
    int64_t out_frame_duration_;

    /* Number of frames that were decoded during seek. */
    uint64_t num_frames_decoded_;

};


// Video Demuxer Interface class
class VideoDemuxer {
    public:
        class StreamProvider {
            public:
                virtual ~StreamProvider() {}
                virtual int GetData(uint8_t *buf, int buf_size) = 0;
        };
        AVCodecID GetCodecID() { return av_video_codec_id_; };
        VideoDemuxer(const char *input_file_path) : VideoDemuxer(CreateFmtContextUtil(input_file_path)) {}
        VideoDemuxer(StreamProvider *stream_provider) : VideoDemuxer(CreateFmtContextUtil(stream_provider)) {av_io_ctx_ = av_fmt_input_ctx_->pb;}
        ~VideoDemuxer();
        bool Demux(uint8_t **video, int *video_size, int64_t *pts = nullptr);
        bool Seek(VideoSeekContext& seek_ctx, uint8_t** pp_video, int* video_size);
        const uint32_t GetWidth() const { return width_;}
        const uint32_t GetHeight() const { return height_;}
        const uint32_t GetChromaHeight() const { return chroma_height_;}
        const uint32_t GetBitDepth() const { return bit_depth_;}
        const uint32_t GetBytePerPixel() const { return byte_per_pixel_;}
        const uint32_t GetBitRate() const { return bit_rate_;}
        const double GetFrameRate() const {return frame_rate_;};
        bool IsVFR() const { return frame_rate_ != avg_frame_rate_; };
        int64_t TsFromTime(double ts_sec) {
            // Convert integer timestamp representation to AV_TIME_BASE and switch to fixed_point
            auto const ts_tbu = llround(ts_sec * AV_TIME_BASE);
            // Rescale the timestamp to value represented in stream base units;
            AVRational time_factor = {1, AV_TIME_BASE};
            return av_rescale_q(ts_tbu, time_factor, av_fmt_input_ctx_->streams[av_stream_]->time_base);
        }

        int64_t TsFromFrameNumber(int64_t frame_num) {
            auto const ts_sec = static_cast<double>(frame_num) / frame_rate_;
            return TsFromTime(ts_sec);
        }

    private:
        VideoDemuxer(AVFormatContext *av_fmt_input_ctx);
        AVFormatContext *CreateFmtContextUtil(StreamProvider *stream_provider);
        AVFormatContext *CreateFmtContextUtil(const char *input_file_path);
        static int ReadPacket(void *data, uint8_t *buf, int buf_size);
        AVFormatContext *av_fmt_input_ctx_ = nullptr;
        AVIOContext *av_io_ctx_ = nullptr;
        AVPacket* packet_ = nullptr;
        AVPacket* packet_filtered_ = nullptr;
        AVBSFContext *av_bsf_ctx_ = nullptr;
        AVCodecID av_video_codec_id_;
        AVPixelFormat chroma_format_;
        double frame_rate_ = 0.0;
        double avg_frame_rate_ = 0.0;
        uint8_t *data_with_header_ = nullptr;
        int av_stream_ = 0;
        bool is_h264_ = false; 
        bool is_hevc_ = false;
        bool is_mpeg4_ = false;
        bool is_seekable_ = false;
        int64_t default_time_scale_ = 1000;
        double time_base_ = 0.0;
        uint32_t frame_count_ = 0;
        uint32_t width_ = 0;
        uint32_t height_ = 0;
        uint32_t chroma_height_ = 0;
        uint32_t bit_depth_ = 0;
        uint32_t byte_per_pixel_ = 0;
        uint32_t bit_rate_ = 0;
        int64_t pkt_dts_ = 0;    // used for Seek Exact frame
};

static inline rocDecVideoCodec AVCodec2RocDecVideoCodec(AVCodecID av_codec) {
    switch (av_codec) {
        case AV_CODEC_ID_MPEG1VIDEO : return rocDecVideoCodec_MPEG1;
        case AV_CODEC_ID_MPEG2VIDEO : return rocDecVideoCodec_MPEG2;
        case AV_CODEC_ID_MPEG4      : return rocDecVideoCodec_MPEG4;
        case AV_CODEC_ID_H264       : return rocDecVideoCodec_AVC;
        case AV_CODEC_ID_HEVC       : return rocDecVideoCodec_HEVC;
        case AV_CODEC_ID_VP8        : return rocDecVideoCodec_VP8;
        case AV_CODEC_ID_VP9        : return rocDecVideoCodec_VP9;
        case AV_CODEC_ID_MJPEG      : return rocDecVideoCodec_JPEG;
        case AV_CODEC_ID_AV1        : return rocDecVideoCodec_AV1;
        default                     : return rocDecVideoCodec_NumCodecs;
    }
}