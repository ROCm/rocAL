# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.10)

project(rocal_video_ffmpeg LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ROCAL_VIDEO_FFMPEG_ADAPTER_STATIC "Build rocal_video_ffmpeg as a static library" OFF)

# FFmpeg is mandatory for the adapter.
find_package(FFmpeg REQUIRED)

# Detect HIP backend independently so the adapter can compile rocDecode support
# even though ENABLE_HIP is only a compile definition of the core target.
set(ROCAL_VIDEO_FFMPEG_HIP OFF)
if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    find_package(HIP QUIET)
    if(HIP_FOUND)
        set(ROCAL_VIDEO_FFMPEG_HIP ON)
    endif()
endif()

# rocDecode is optional and only required on HIP builds.
set(ROCAL_VIDEO_FFMPEG_ROCDECODE OFF)
if(ROCAL_VIDEO_FFMPEG_HIP)
    find_package(rocdecode 1.0.0 QUIET)
    if(rocdecode_FOUND)
        set(ROCAL_VIDEO_FFMPEG_ROCDECODE ON)
    endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/rocAL/include
    ${CMAKE_SOURCE_DIR}/rocAL/include/api
    ${CMAKE_BINARY_DIR}/rocAL
    ${AVUTIL_INCLUDE_DIR}
    ${AVCODEC_INCLUDE_DIR}
    ${SWSCALE_INCLUDE_DIR}
    ${AVFORMAT_INCLUDE_DIR}
)

set(ROCAL_VIDEO_FFMPEG_SOURCES
    source/register.cpp
    source/ffmpeg_video_decoder.cpp
    source/video_properties.cpp
)

if(ROCAL_VIDEO_FFMPEG_HIP AND ROCAL_VIDEO_FFMPEG_ROCDECODE)
    list(APPEND ROCAL_VIDEO_FFMPEG_SOURCES source/rocdec_video_decoder.cpp)
    include_directories(
        ${ROCM_PATH}/include
        ${rocdecode_INCLUDE_DIR}
        ${ROCM_PATH}/share/rocdecode/utils/rocvideodecode
        ${ROCM_PATH}/share/rocdecode/utils
    )
endif()

if(ROCAL_VIDEO_FFMPEG_ADAPTER_STATIC)
    add_library(${PROJECT_NAME} STATIC ${ROCAL_VIDEO_FFMPEG_SOURCES})
else()
    add_library(${PROJECT_NAME} SHARED ${ROCAL_VIDEO_FFMPEG_SOURCES})
endif()

# FFMPEG multi-version support
if(_FFMPEG_AVCODEC_VERSION VERSION_LESS_EQUAL 58.134.100)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_AVCODEC_GREATER_THAN_58_134=0)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_AVCODEC_GREATER_THAN_58_134=1)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE ROCAL_VIDEO)
if(ROCAL_VIDEO_FFMPEG_HIP)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_HIP=1 __HIP_PLATFORM_AMD__)
    if(ROCAL_VIDEO_FFMPEG_ROCDECODE)
        target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_ROCDECODE)
    endif()
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_HIP=0)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC
    rocal
    ${FFMPEG_LIBRARIES}
)

if(ROCAL_VIDEO_FFMPEG_HIP AND ROCAL_VIDEO_FFMPEG_ROCDECODE)
    target_link_libraries(${PROJECT_NAME} PUBLIC rocdecode::rocdecode)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${rocal_VERSION}
    SOVERSION ${rocal_VERSION_MAJOR}
)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev
)
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY
)

# Install adapter sources under COMPONENT dev (issue comment #2).
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.cpp"
    PATTERN "*.hpp"
    PATTERN "CMakeLists.txt"
)
