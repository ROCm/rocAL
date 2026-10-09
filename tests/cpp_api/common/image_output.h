/*
MIT License

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Show or save an image for the cpp_api tests.
// This is the only place that checks OpenCV and highgui.
// A test asks to show or save an image. DISPLAY in this header chooses which.
// When it is 1 and highgui is available, the image is shown. Otherwise it is written.
// Replacing OpenCV with another utility means changing this file only.

#pragma once

// 0 writes the image to a file. 1 requests a window when highgui is available.
#define DISPLAY 0

#include <string>

#if ENABLE_OPENCV
#include <opencv2/opencv.hpp>
#include <vector>

// displayed is what a window shows. written is what a file receives.
// They are the same Mat for most tests. png_compression < 0 keeps OpenCV's default.
// wait_ms < 0 does not wait after a window update.
inline void rocal_test_output_image(const cv::Mat& displayed,
                                     const cv::Mat& written,
                                     const std::string& path,
                                     const char* window = "output",
                                     int wait_ms = -1,
                                     int png_compression = -1) {
#if ENABLE_OPENCV_HIGHGUI
    if (DISPLAY) {
        cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
        cv::imshow(window, displayed);
        if (wait_ms >= 0)
            cv::waitKey(wait_ms);
        return;
    }
#else
    (void)displayed;
    (void)window;
    (void)wait_ms;
#endif
    if (png_compression >= 0) {
        const std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, png_compression};
        cv::imwrite(path, written, params);
    } else {
        cv::imwrite(path, written);
    }
}

// Wait only when a window was requested and highgui is available.
inline void rocal_test_output_wait(int wait_ms) {
#if ENABLE_OPENCV_HIGHGUI
    if (DISPLAY)
        cv::waitKey(wait_ms);
#else
    (void)wait_ms;
#endif
}

// Pause when highgui is available, whether or not a window was requested.
inline void rocal_test_output_pause(int wait_ms) {
#if ENABLE_OPENCV_HIGHGUI
    cv::waitKey(wait_ms);
#else
    (void)wait_ms;
#endif
}

inline void rocal_test_output_open(const char* window) {
#if ENABLE_OPENCV_HIGHGUI
    if (DISPLAY)
        cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
#else
    (void)window;
#endif
}

inline void rocal_test_output_close(const char* window) {
#if ENABLE_OPENCV_HIGHGUI
    if (DISPLAY)
        cv::destroyWindow(window);
#else
    (void)window;
#endif
}

#endif  // ENABLE_OPENCV
