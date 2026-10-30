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

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# Default backend options when this CMake is invoked standalone from installed sources.
if(NOT DEFINED GPU_SUPPORT)
    option(GPU_SUPPORT "Build with GPU support" ON)
endif()
if(NOT DEFINED BACKEND)
    set(BACKEND "HIP" CACHE STRING "Backend [CPU/HIP]")
endif()

# Detect whether we are building inside the full rocAL source tree or standalone
# from the sources shipped with the rocAL dev package.
if(EXISTS ${CMAKE_SOURCE_DIR}/rocAL/include)
    set(ROCAL_VIDEO_FFMPEG_IN_TREE ON)
else()
    set(ROCAL_VIDEO_FFMPEG_IN_TREE OFF)
endif()

if(ROCAL_VIDEO_FFMPEG_IN_TREE)
    message("-- ${White}rocal_video_ffmpeg: building in-tree${ColourReset}")
    set(ROCAL_TARGET rocal)
    set(ROCAL_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/rocAL/include
        ${CMAKE_SOURCE_DIR}/rocAL/include/api
        ${CMAKE_BINARY_DIR}/rocAL
    )
    if(MIVisionX_INCLUDE_DIRS)
        list(APPEND ROCAL_INCLUDE_DIRS ${MIVisionX_INCLUDE_DIRS})
    endif()
    list(APPEND ROCAL_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/rocAL/include)
else()
    message("-- ${White}rocal_video_ffmpeg: building standalone from installed sources${ColourReset}")
    # Allow standalone builds to find rocAL's custom CMake modules (e.g. FindFFmpeg.cmake).
    list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rocal/cmake)
    # Find the installed rocAL library.
    find_library(ROCAL_LIBRARY NAMES rocal PATHS ${ROCM_PATH}/lib ${CMAKE_INSTALL_PREFIX}/lib)
    if(NOT ROCAL_LIBRARY)
        message(FATAL_ERROR "-- ${Red}rocal_video_ffmpeg: rocAL library not found under ${ROCM_PATH}/lib or ${CMAKE_INSTALL_PREFIX}/lib${ColourReset}")
    endif()
    add_library(rocal_imported SHARED IMPORTED)
    set_target_properties(rocal_imported PROPERTIES IMPORTED_LOCATION ${ROCAL_LIBRARY})
    set(ROCAL_TARGET rocal_imported)
    # rocAL dev package install headers under <prefix>/include/rocal but internal
    # headers include each other with relative paths like "decoders/video/..." or
    # "readers/video/...". Add both the prefix and the rocal subdirectory so the
    # adapter can find them without rewriting all include statements.
    set(ROCAL_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ROCM_PATH}/include
        ${ROCM_PATH}/include/rocal
        ${CMAKE_INSTALL_PREFIX}/include/rocal
    )
    find_path(MIVisionX_INCLUDE_DIR NAMES vx_ext_amd.h PATHS ${MIVisionX_PATH}/include/mivisionx ${ROCM_PATH}/include/mivisionx NO_DEFAULT_PATH)
    if(MIVisionX_INCLUDE_DIR)
        list(APPEND ROCAL_INCLUDE_DIRS ${MIVisionX_INCLUDE_DIR})
    endif()
endif()

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
    ${ROCAL_INCLUDE_DIRS}
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
    ${ROCAL_TARGET}
    ${FFMPEG_LIBRARIES}
)

if(ROCAL_VIDEO_FFMPEG_HIP AND ROCAL_VIDEO_FFMPEG_ROCDECODE)
    target_link_libraries(${PROJECT_NAME} PUBLIC rocdecode::rocdecode)
endif()

if(NOT DEFINED rocal_VERSION)
    set(rocal_VERSION "2.5.0")
endif()
if(NOT DEFINED rocal_VERSION_MAJOR)
    string(REGEX MATCH "^([0-9]+)" rocal_VERSION_MAJOR "${rocal_VERSION}")
endif()

# GNUInstallDirs is required for the standalone install layout below.
include(GNUInstallDirs)

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
    DESTINATION ${CMAKE_INSTALL_DATADIR}/rocal/rocal_video_ffmpeg
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.cpp"
    PATTERN "*.hpp"
    PATTERN "CMakeLists.txt"
)
