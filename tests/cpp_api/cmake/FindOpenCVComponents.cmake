################################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

# Image output (core, imgproc, imgcodecs) and display (highgui) are separate.
# A headless OpenCV build still provides cv::Mat, cvtColor, and imwrite.
# tests/cpp_api/common holds image_output.h, the shared show-or-save helper.
function(rocal_test_enable_opencv target)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")

    find_package(OpenCV QUIET)

    set(_io_ok FALSE)
    set(_highgui_ok FALSE)
    if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER_EQUAL 4)
        if("opencv_core" IN_LIST OpenCV_LIB_COMPONENTS
           AND "opencv_imgproc" IN_LIST OpenCV_LIB_COMPONENTS
           AND "opencv_imgcodecs" IN_LIST OpenCV_LIB_COMPONENTS)
            set(_io_ok TRUE)
        endif()
        if("opencv_highgui" IN_LIST OpenCV_LIB_COMPONENTS)
            set(_highgui_ok TRUE)
        endif()
    endif()

    if(_io_ok)
        message(STATUS "${target}: OpenCV ${OpenCV_VERSION} image output enabled")
        target_compile_definitions(${target} PUBLIC ENABLE_OPENCV=1)
        target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${target} ${OpenCV_LIBRARIES})
    else()
        message(STATUS "${target}: OpenCV image output disabled")
        target_compile_definitions(${target} PUBLIC ENABLE_OPENCV=0)
    endif()

    if(_highgui_ok)
        message(STATUS "${target}: OpenCV highgui enabled")
        target_compile_definitions(${target} PUBLIC ENABLE_OPENCV_HIGHGUI=1)
    else()
        message(STATUS "${target}: OpenCV highgui disabled; display calls compiled out")
        target_compile_definitions(${target} PUBLIC ENABLE_OPENCV_HIGHGUI=0)
    endif()

    # Visible to find_package. False is a valid build: image output or highgui may be absent.
    set(OpenCVComponents_FOUND ${_io_ok} PARENT_SCOPE)
    set(OpenCVComponents_HIGHGUI_FOUND ${_highgui_ok} PARENT_SCOPE)
endfunction()

# Loaded by find_package after add_executable(), so PROJECT_NAME is that test.
rocal_test_enable_opencv(${PROJECT_NAME})
message(STATUS "OpenCVComponents_FOUND=${OpenCVComponents_FOUND}")
message(STATUS "OpenCVComponents_HIGHGUI_FOUND=${OpenCVComponents_HIGHGUI_FOUND}")
