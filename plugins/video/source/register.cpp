/**
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

#include "rocal_video_plugin/register.h"

#include <iostream>

#include "decoders/video/video_decoder_registry.h"
#include "decoders/video/video_plugin_loader.h"
#include "readers/video/video_properties.h"

#include "rocal_video_plugin/ffmpeg_video_decoder.h"

#if ENABLE_HIP && ENABLE_ROCDECODE
#include "rocal_video_plugin/rocdec_video_decoder.h"
#endif

// Implemented in video_properties.cpp and registered into the core registry.
extern void ffmpeg_open_video_context(const char *video_file_path, Properties &props);
extern void ffmpeg_get_video_properties_from_txt_file(VideoProperties &video_props, const char *file_path, bool file_list_frame_num);
extern void ffmpeg_find_video_properties(VideoProperties &video_props, const char *source_path, bool file_list_frame_num);

namespace rocal {
namespace video {

namespace {

class VideoPluginRegistrar {
   public:
    VideoPluginRegistrar() {
        if (rocal_video_plugin_core_enable_hip() != ENABLE_HIP) {
            std::cerr << "rocal_video_plugin: built with ENABLE_HIP=" << ENABLE_HIP
                      << " but librocal was built with ENABLE_HIP=" << rocal_video_plugin_core_enable_hip()
                      << "; not registering video decoders. Rebuild rocal_video_plugin against this rocAL.\n";
            return;
        }
        register_decoder_factory(DecoderType::FFMPEG_VIDEO,
                                 [](DecoderConfig& cfg) { return std::make_shared<FFmpegVideoDecoder>(); });
#if ENABLE_HIP && ENABLE_ROCDECODE
        register_decoder_factory(DecoderType::ROCDECODE_VIDEO,
                                 [](DecoderConfig& cfg) { return std::make_shared<RocDecVideoDecoder>(cfg.get_hip_stream()); });
#endif
        register_video_properties_functions(
            [](const char *path, Properties &props) { ffmpeg_open_video_context(path, props); },
            [](VideoProperties &video_props, const char *file_path, bool flag) { ffmpeg_get_video_properties_from_txt_file(video_props, file_path, flag); },
            [](VideoProperties &video_props, const char *source_path, bool flag) { ffmpeg_find_video_properties(video_props, source_path, flag); });
    }
};

// Static instance in the plugin .so triggers auto-registration on load.
static VideoPluginRegistrar g_video_plugin_registrar;

}  // namespace

void register_ffmpeg_decoders() {
    // The static registrar has already performed registration; this function is
    // provided for users that prefer an explicit call.
}

}  // namespace video
}  // namespace rocal

// C-compatible symbol that core can look up when loading the plugin at runtime.
extern "C" void rocal_video_plugin_register() {
    rocal::video::register_ffmpeg_decoders();
}
