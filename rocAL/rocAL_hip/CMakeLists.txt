# Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.5)
project(rocAL_hip CXX)

find_package(rocDecode QUIET)

list(APPEND rocAL_HIP_SOURCES
    rocal_hip_kernels.cpp
    box_encoder_hip.cpp
    utils/rocvideodecode/roc_video_dec.cpp
    utils/colorspace_kernels.cpp
    utils/resize_kernels.cpp
    utils/video_demuxer.cpp
)

if(NOT rocDecode_FOUND)
    message("-- ${Yellow}NOTE: rocAL built without rocDecode Video Hardware Decode Functionality${ColourReset}")
elseif(HIP_FOUND AND FFMPEG_FOUND)
    include_directories (${rocDecode_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${rocDecode_LIBRARIES})
    message("-- ${White}rocAL built with rocDecode Video Decode Functionality${ColourReset}")
else()
    message("-- ${Yellow}NOTE: rocAL built without rocDecode Video Hardware Decode Functionality as FFmepg not found${ColourReset}")
endif()

set_source_files_properties(${rocAL_HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

set(HIP_CXX_FLAGS -std=gnu++17)
set(COMPILER_FOR_HIP ${ROCM_PATH}/llvm/bin/clang++)
set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CXX_FLAGS}")

add_library(${PROJECT_NAME} OBJECT ${rocAL_HIP_SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(${PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries(${PROJECT_NAME} hip::device)
