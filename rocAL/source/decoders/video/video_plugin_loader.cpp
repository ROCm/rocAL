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
#include <functional>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <unordered_set>
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

struct OpenedPlugin {
    dev_t device;
    ino_t inode;
    bool operator==(const OpenedPlugin &other) const {
        return device == other.device && inode == other.inode;
    }
};

struct OpenedPluginHash {
    size_t operator()(const OpenedPlugin &plugin) const {
        return std::hash<unsigned long long>{}(static_cast<unsigned long long>(plugin.device)) ^
               (std::hash<unsigned long long>{}(static_cast<unsigned long long>(plugin.inode)) << 1);
    }
};

bool plugin_file_id(const std::string &path, OpenedPlugin &id) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
        return false;
    id = OpenedPlugin{info.st_dev, info.st_ino};
    return true;
}

// Identity of the file dlopen actually mapped. stat follows symlinks, so a
// versioned SONAME and its .so symlink share one id.
bool loaded_plugin_id(void *handle, const std::string &candidate, OpenedPlugin &id) {
    struct link_map *map = nullptr;
    if (dlinfo(handle, RTLD_DI_LINKMAP, &map) == 0 && map != nullptr && map->l_name != nullptr && map->l_name[0] != '\0') {
        if (plugin_file_id(map->l_name, id))
            return true;
    }
    if (candidate.find('/') != std::string::npos)
        return plugin_file_id(candidate, id);
    return false;
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
    std::unordered_set<std::string> tried_names;
    std::unordered_set<OpenedPlugin, OpenedPluginHash> opened_files;
    // Kept mapped until the search ends. dlclose'ing a rejected plugin before the
    // next candidate lets a later name for that same file (the .so symlink, or the
    // bare SONAME) load it again and rerun its constructor.
    std::vector<void *> rejected;
    for (const auto &candidate : build_plugin_search_list()) {
        if (!tried_names.insert(candidate).second)
            continue;
        if (candidate.find('/') != std::string::npos) {
            OpenedPlugin id;
            if (plugin_file_id(candidate, id) && opened_files.find(id) != opened_files.end())
                continue;
        }

        void *handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            OpenedPlugin id;
            if (loaded_plugin_id(handle, candidate, id) && !opened_files.insert(id).second) {
                dlclose(handle);  // extra reference to a file already opened
                continue;
            }
            if (is_decoder_registered(DecoderType::FFMPEG_VIDEO)) {
                for (void *rejected_handle : rejected)
                    dlclose(rejected_handle);
                return;  // plugin constructor registered the factories.
            }
            // The .so loaded but did not register; keep it mapped and keep searching.
            if (!error_log.empty()) error_log += "; ";
            error_log += candidate + ": loaded but registered no decoders (built against a different rocAL backend?)";
            rejected.push_back(handle);
        } else {
            const char *err = dlerror();
            if (err) {
                if (!error_log.empty()) error_log += "; ";
                error_log += candidate + ": " + err;
            }
        }
    }

    for (void *rejected_handle : rejected)
        dlclose(rejected_handle);

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
