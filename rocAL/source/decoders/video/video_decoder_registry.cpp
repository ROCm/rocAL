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

#include "decoders/video/video_decoder_registry.h"

#include <map>
#include <mutex>
#include "pipeline/commons.h"

namespace rocal {
namespace video {

namespace {

static std::map<DecoderType, VideoDecoderFactory> g_decoder_factories;
static std::mutex g_decoder_registry_mutex;

}  // namespace

void register_decoder_factory(DecoderType type, VideoDecoderFactory factory) {
    std::lock_guard<std::mutex> lock(g_decoder_registry_mutex);
    g_decoder_factories[type] = std::move(factory);
}

std::shared_ptr<VideoDecoder> create_decoder(DecoderType type, const DecoderConfig& cfg) {
    std::lock_guard<std::mutex> lock(g_decoder_registry_mutex);
    auto it = g_decoder_factories.find(type);
    if (it == g_decoder_factories.end() || !it->second)
        THROW("No video decoder factory is registered for decoder type " + TOSTR(type) +
              ". Link the rocal_video_ffmpeg adapter if video support is required.");
    return it->second(cfg);
}

bool is_decoder_registered(DecoderType type) {
    std::lock_guard<std::mutex> lock(g_decoder_registry_mutex);
    auto it = g_decoder_factories.find(type);
    return it != g_decoder_factories.end() && static_cast<bool>(it->second);
}

}  // namespace video
}  // namespace rocal
