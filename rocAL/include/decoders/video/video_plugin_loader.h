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

#pragma once

namespace rocal {
namespace video {

// Ensures the optional rocal_video_plugin shared library is loaded so that
// FFmpeg/rocDecode video decoder factories and video property functions are
// registered.  It is a no-op when the plugin has already been linked in or when
// rocAL was built in monolithic mode (the factories are registered statically
// inside librocal.so).
void ensure_video_plugin_loaded();

}  // namespace video
}  // namespace rocal

// ENABLE_HIP value librocal was built with. The plugin compares it with its own value before
// registering, because rocAL's internal types (e.g. DecoderConfig) differ between backends.
extern "C" int rocal_video_plugin_core_enable_hip();
