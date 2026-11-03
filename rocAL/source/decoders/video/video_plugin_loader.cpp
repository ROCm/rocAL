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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "decoders/video/video_plugin_loader.h"

#include "decoders/video/video_decoder_registry.h"

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "pipeline/commons.h"

#ifdef ROCAL_VIDEO_PLUGIN
#include <dlfcn.h>
#endif

#ifndef ROCM_PATH
#define ROCM_PATH "/opt/rocm"
#endif

namespace rocal {
namespace video {

#ifdef ROCAL_VIDEO_PLUGIN

namespace {

const char kPluginName[] = "librocal_video_plugin.so";
const char kPluginNameVersioned[] = "librocal_video_plugin.so.2";

// Append a directory and both the versioned and unversioned plugin names to the
// search list.  Runtime packages use NAMELINK_SKIP, so the .so symlink may be
// absent and only the .so.2 SONAME is present.
void append_plugin_candidates(std::vector<std::string> &paths, const std::string &dir) {
    if (dir.empty()) return;
    paths.emplace_back(dir + "/" + kPluginNameVersioned);
    paths.emplace_back(dir + "/" + kPluginName);
}

// Return the directory containing the running librocal.so so we can look for
// the plugin next to it.  If dladdr fails, return an empty string.
std::string get_rocal_library_dir() {
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(&get_rocal_library_dir), &info) && info.dli_fname) {
        const char *slash = strrchr(info.dli_fname, '/');
        if (slash) {
            return std::string(info.dli_fname, slash - info.dli_fname);
        }
    }
    return {};
}

std::vector<std::string> build_plugin_search_list() {
    std::vector<std::string> paths;
    // 1. Let the dynamic linker honour LD_LIBRARY_PATH / rpath by trying the
    //    bare and versioned SONAMEs first.
    paths.emplace_back(kPluginNameVersioned);
    paths.emplace_back(kPluginName);
    // 2. Directory that contains librocal.so.
    append_plugin_candidates(paths, get_rocal_library_dir());
    // 3. ROCm default install path.
    append_plugin_candidates(paths, std::string(ROCM_PATH) + "/lib");
    return paths;
}

void do_load_video_plugin() {
    if (is_decoder_registered(DecoderType::FFMPEG_VIDEO)) {
        return;  // already loaded, e.g. app linked it.
    }

    std::string error_log;
    for (const auto &candidate : build_plugin_search_list()) {
        void *handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            if (is_decoder_registered(DecoderType::FFMPEG_VIDEO)) {
                return;  // plugin constructor registered the factories.
            }
            // The .so loaded but did not register; close it and keep searching.
            dlclose(handle);
        } else {
            const char *err = dlerror();
            if (err) {
                if (!error_log.empty()) error_log += "; ";
                error_log += candidate + ": " + err;
            }
        }
    }

    THROW("rocAL video plugin (" + std::string(kPluginName) + " or " +
          std::string(kPluginNameVersioned) +
          ") could not be loaded. Ensure it is installed next to librocal.so or in "
          "the ROCm library path (" + std::string(ROCM_PATH) +
          "/lib). If the host FFmpeg/rocDecode libraries are incompatible, rebuild "
          "rocal_video_plugin against the host libraries. Details: " + error_log);
}

std::once_flag g_plugin_load_once;

}  // namespace

void ensure_video_plugin_loaded() {
    std::call_once(g_plugin_load_once, do_load_video_plugin);
}

#else  // !ROCAL_VIDEO_PLUGIN

void ensure_video_plugin_loaded() {
    // No-op in monolithic builds: the decoder factories are part of librocal.so.
}

#endif  // ROCAL_VIDEO_PLUGIN

}  // namespace video
}  // namespace rocal
