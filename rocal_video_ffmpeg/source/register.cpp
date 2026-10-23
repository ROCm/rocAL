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

#include "rocal_video_ffmpeg/register.h"

#include "decoders/video/video_decoder_registry.h"
#include "readers/video/video_properties.h"

#include "rocal_video_ffmpeg/ffmpeg_video_decoder.h"

#if ENABLE_HIP && ENABLE_ROCDECODE
#include "rocal_video_ffmpeg/rocdec_video_decoder.h"
#endif

namespace rocal {
namespace video {

namespace {

class FFmpegAdapterRegistrar {
   public:
    FFmpegAdapterRegistrar() {
        register_decoder_factory(DecoderType::FFMPEG_VIDEO,
                                 [](DecoderConfig& cfg) { return std::make_shared<FFmpegVideoDecoder>(); });
#if ENABLE_HIP && ENABLE_ROCDECODE
        register_decoder_factory(DecoderType::ROCDECODE_VIDEO,
                                 [](DecoderConfig& cfg) { return std::make_shared<RocDecVideoDecoder>(cfg.get_hip_stream()); });
#endif
        register_video_properties_functions(
            [](const char *path, Properties &props) { rocal_video_ffmpeg_open_video_context(path, props); },
            [](VideoProperties &video_props, const char *file_path, bool flag) { rocal_video_ffmpeg_get_video_properties_from_txt_file(video_props, file_path, flag); },
            [](VideoProperties &video_props, const char *source_path, bool flag) { rocal_video_ffmpeg_find_video_properties(video_props, source_path, flag); });
    }
};

// Static instance in the adapter .so triggers auto-registration on load.
static FFmpegAdapterRegistrar g_ffmpeg_adapter_registrar;

}  // namespace

void register_ffmpeg_decoders() {
    // The static registrar has already performed registration; this function is
    // provided for users that prefer an explicit call.
}

}  // namespace video
}  // namespace rocal

// C-compatible symbols that core can look up if needed.
extern "C" void rocal_video_ffmpeg_register() {
    rocal::video::register_ffmpeg_decoders();
}
