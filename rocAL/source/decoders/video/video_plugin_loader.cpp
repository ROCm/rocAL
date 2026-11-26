/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include <set>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

#ifdef ROCAL_VIDEO_PLUGIN
#include <link.h>
#endif

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

#if !defined(ROCAL_VIDEO_PLUGIN_FILENAME) || !defined(ROCAL_VIDEO_PLUGIN_SONAME)
#error "ROCAL_VIDEO_PLUGIN_FILENAME and ROCAL_VIDEO_PLUGIN_SONAME must be defined by the build"
#endif

const char kPluginName[] = ROCAL_VIDEO_PLUGIN_FILENAME;
const char kPluginNameVersioned[] = ROCAL_VIDEO_PLUGIN_SONAME;

// Append a directory and both the versioned and unversioned plugin names to the
// search list.  Runtime packages use NAMELINK_SKIP, so the .so symlink may be
// absent and only the versioned SONAME is present.
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

using FileId = std::pair<dev_t, ino_t>;

// stat follows symlinks, so a versioned SONAME and its .so symlink share one id.
bool file_id_of(const char *path, FileId &id) {
    struct stat info;
    if (stat(path, &info) != 0)
        return false;
    id = {info.st_dev, info.st_ino};
    return true;
}

// Identity of the file dlopen actually mapped, which is how a bare SONAME is resolved.
bool mapped_file_id(void *handle, FileId &id) {
    struct link_map *map = nullptr;
    return dlinfo(handle, RTLD_DI_LINKMAP, &map) == 0 && map != nullptr && map->l_name != nullptr &&
           map->l_name[0] != '\0' && file_id_of(map->l_name, id);
}

std::vector<std::string> build_plugin_search_list() {
    std::vector<std::string> paths;
    // 1. Directory that contains the running librocal.so. The co-located plugin shares this
    //    exact build's internal C++ types with the core, so it must take precedence over any
    //    other copy the dynamic linker could resolve via LD_LIBRARY_PATH / rpath.
    append_plugin_candidates(paths, get_rocal_library_dir());
    // 2. Fall back to the bare and versioned SONAMEs so the dynamic linker can honour
    //    LD_LIBRARY_PATH / rpath / the default library search path.
    paths.emplace_back(kPluginNameVersioned);
    paths.emplace_back(kPluginName);
    // 3. ROCm default install path.
    append_plugin_candidates(paths, std::string(ROCM_PATH) + "/lib");
    return paths;
}

void do_load_video_plugin() {
    if (is_decoder_registered(DecoderType::FFMPEG_VIDEO)) {
        return;  // already loaded, e.g. app linked it.
    }

    std::string error_log;
    std::set<std::string> tried_names;
    // A rejected plugin is closed at once: while it stays loaded, a later bare-SONAME
    // dlopen returns it instead of searching LD_LIBRARY_PATH / rpath. Its file id is
    // kept so another path to the same file is not opened again.
    std::set<FileId> rejected_files;
    for (const auto &candidate : build_plugin_search_list()) {
        if (!tried_names.insert(candidate).second)
            continue;
        FileId id;
        if (candidate.find('/') != std::string::npos && file_id_of(candidate.c_str(), id) &&
            rejected_files.count(id) != 0)
            continue;

        // RTLD_LOCAL: a rejected plugin must not add its symbols to the global scope,
        // where they would bind ahead of a later plugin's own definitions.
        void *handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            const char *err = dlerror();
            if (err) {
                if (!error_log.empty()) error_log += "; ";
                error_log += candidate + ": " + err;
            }
            continue;
        }
        if (is_decoder_registered(DecoderType::FFMPEG_VIDEO)) {
            dlopen(candidate.c_str(), RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
            return;  // plugin constructor registered the factories.
        }
        // The .so loaded but did not register; close it and keep searching.
        FileId mapped;
        const bool seen = mapped_file_id(handle, mapped) && !rejected_files.insert(mapped).second;
        if (!seen) {
            if (!error_log.empty()) error_log += "; ";
            error_log += candidate + ": loaded but registered no decoders (built against a different rocAL backend?)";
        }
        dlclose(handle);
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

extern "C" int rocal_video_plugin_core_enable_hip() {
    return ENABLE_HIP;
}
