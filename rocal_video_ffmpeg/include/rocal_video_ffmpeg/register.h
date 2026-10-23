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

#pragma once

#include "readers/video/video_properties.h"

namespace rocal {
namespace video {

// Adapter-provided property functions (also declared in core's video_properties.h
// via the canonical names).  The adapter implementation uses these names internally.
void ffmpeg_open_video_context(const char *video_file_path, Properties &props);
void ffmpeg_get_video_properties_from_txt_file(VideoProperties &video_props, const char *file_path, bool file_list_frame_num);
void ffmpeg_find_video_properties(VideoProperties &video_props, const char *source_path, bool file_list_frame_num);

// Register FFmpeg and (optionally) rocDecode video decoders and video properties
// functions into the core rocAL registries.  Calling this function is optional:
// linking against the rocal_video_ffmpeg shared library also triggers
// registration automatically through a constructor-side helper.
void register_ffmpeg_decoders();

}  // namespace video
}  // namespace rocal

// The adapter implementation in video_properties.cpp defines the property helpers
// in the global namespace.  Provide rocal::video aliases for them so that
// register.cpp can use the namespaced form while video_properties.cpp keeps the
// global form.  This avoids a link mismatch where register.cpp calls
// rocal::video::ffmpeg_* but the .so only exports the undecorated symbols.
extern "C++" {
void ffmpeg_open_video_context(const char *video_file_path, Properties &props);
void ffmpeg_get_video_properties_from_txt_file(VideoProperties &video_props, const char *file_path, bool file_list_frame_num);
void ffmpeg_find_video_properties(VideoProperties &video_props, const char *source_path, bool file_list_frame_num);
}

namespace rocal {
namespace video {
inline void rocal_video_ffmpeg_open_video_context(const char *video_file_path, Properties &props) { ::ffmpeg_open_video_context(video_file_path, props); }
inline void rocal_video_ffmpeg_get_video_properties_from_txt_file(VideoProperties &video_props, const char *file_path, bool file_list_frame_num) { ::ffmpeg_get_video_properties_from_txt_file(video_props, file_path, file_list_frame_num); }
inline void rocal_video_ffmpeg_find_video_properties(VideoProperties &video_props, const char *source_path, bool file_list_frame_num) { ::ffmpeg_find_video_properties(video_props, source_path, file_list_frame_num); }
}  // namespace video
}  // namespace rocal

// C-compatible symbol that core can look up if needed.
extern "C" void rocal_video_ffmpeg_register();
