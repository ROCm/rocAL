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

#include "readers/video/video_properties.h"

#include <mutex>

namespace {

using OpenVideoContextFn = void (*)(const char *, Properties &);
using FindVideoPropertiesFn = void (*)(VideoProperties &, const char *, bool);
using GetVideoPropertiesFromTxtFileFn = void (*)(VideoProperties &, const char *, bool);

struct PropertiesRegistry {
    OpenVideoContextFn open_video_context = nullptr;
    FindVideoPropertiesFn find_video_properties = nullptr;
    GetVideoPropertiesFromTxtFileFn get_video_properties_from_txt_file = nullptr;
};

PropertiesRegistry & properties_registry() {
    static PropertiesRegistry registry;
    return registry;
}

std::mutex & properties_registry_mutex() {
    static std::mutex mtx;
    return mtx;
}

}  // namespace

void substring_extraction(std::string const &str, const char delim, std::vector<std::string> &out) {
    size_t start;
    size_t end = 0;
    while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}

void register_video_properties_functions(
    void (*open_video_context_fn)(const char *, Properties &),
    void (*get_video_properties_from_txt_file_fn)(VideoProperties &, const char *, bool),
    void (*find_video_properties_fn)(VideoProperties &, const char *, bool)) {
    std::lock_guard<std::mutex> lock(properties_registry_mutex());
    properties_registry().open_video_context = open_video_context_fn;
    properties_registry().get_video_properties_from_txt_file = get_video_properties_from_txt_file_fn;
    properties_registry().find_video_properties = find_video_properties_fn;
}

void open_video_context(const char *video_file_path, Properties &props) {
    std::lock_guard<std::mutex> lock(properties_registry_mutex());
    if (!properties_registry().open_video_context)
        THROW("No video properties adapter is registered. Link rocal_video_ffmpeg if video support is required.");
    properties_registry().open_video_context(video_file_path, props);
}

void get_video_properties_from_txt_file(VideoProperties &video_props, const char *file_path, bool file_list_frame_num) {
    std::lock_guard<std::mutex> lock(properties_registry_mutex());
    if (!properties_registry().get_video_properties_from_txt_file)
        THROW("No video properties adapter is registered. Link rocal_video_ffmpeg if video support is required.");
    properties_registry().get_video_properties_from_txt_file(video_props, file_path, file_list_frame_num);
}

void find_video_properties(VideoProperties &video_props, const char *source_path, bool file_list_frame_num) {
    std::lock_guard<std::mutex> lock(properties_registry_mutex());
    if (!properties_registry().find_video_properties)
        THROW("No video properties adapter is registered. Link rocal_video_ffmpeg if video support is required.");
    properties_registry().find_video_properties(video_props, source_path, file_list_frame_num);
}
