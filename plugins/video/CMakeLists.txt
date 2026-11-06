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

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# The rocDecode color-conversion and resize kernels are compiled into the plugin, which needs a
# HIP-capable compiler. Standalone builds would otherwise pick the host default (usually g++).
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

project(rocal_video_plugin LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ROCAL_VIDEO_PLUGIN_STATIC "Build rocal_video_plugin as a static library" OFF)

# Detect whether we are building inside the full rocAL source tree or standalone
# from the sources shipped with the rocAL dev package.
if(EXISTS ${CMAKE_SOURCE_DIR}/rocAL/include)
    set(ROCAL_VIDEO_PLUGIN_IN_TREE ON)
else()
    set(ROCAL_VIDEO_PLUGIN_IN_TREE OFF)
endif()

if(ROCAL_VIDEO_PLUGIN_IN_TREE)
    message("-- ${White}rocal_video_plugin: building in-tree${ColourReset}")
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
    message("-- ${White}rocal_video_plugin: building standalone from installed sources${ColourReset}")
    # librocal dlopen()s the plugin, and the plugin's own RUNPATH is what resolves its ROCm
    # dependencies (e.g. librocdecode). Without it, tarball ROCm installs that are not
    # registered with ldconfig fail to load the plugin. $ORIGIN covers the plugin installed next
    # to librocal in the ROCm lib directory, and ROCM_PATH/lib covers a separate install prefix.
    if(NOT DEFINED CMAKE_INSTALL_RPATH)
        set(CMAKE_INSTALL_RPATH "\$ORIGIN;${ROCM_PATH}/lib")
    endif()
    # Make ROCm packages (HIP, rocDecode) discoverable without caller-supplied hints,
    # matching rocAL/CMakeLists.txt. This must run before the HIP/rocDecode discovery below.
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/lib/cmake ${ROCM_PATH}/hip)
    # Allow standalone builds to find rocAL's custom CMake modules (e.g. FindFFmpeg.cmake).
    # rocAL may have been installed either under ROCM_PATH or a separate CMAKE_INSTALL_PREFIX
    # (the rocAL library below is searched in both), so look for the modules in both prefixes.
    list(APPEND CMAKE_MODULE_PATH
        ${ROCM_PATH}/share/rocal/cmake
        ${CMAKE_INSTALL_PREFIX}/share/rocal/cmake)
    # Find the installed rocAL library.
    find_library(ROCAL_LIBRARY NAMES rocal PATHS ${ROCM_PATH}/lib ${CMAKE_INSTALL_PREFIX}/lib)
    if(NOT ROCAL_LIBRARY)
        message(FATAL_ERROR "-- ${Red}rocal_video_plugin: rocAL library not found under ${ROCM_PATH}/lib or ${CMAKE_INSTALL_PREFIX}/lib${ColourReset}")
    endif()
    add_library(rocal_imported SHARED IMPORTED)
    set_target_properties(rocal_imported PROPERTIES IMPORTED_LOCATION ${ROCAL_LIBRARY})
    set(ROCAL_TARGET rocal_imported)
    # Resolve the rocAL headers from the SAME installation prefix that provided the selected
    # library (its parent of <prefix>/lib), so the plugin compiles against the matching
    # VideoDecoder interface instead of possibly-older headers from a different rocAL prefix.
    # Internal headers include each other with relative paths like "decoders/video/..." or
    # "readers/video/...", so add both <prefix>/include and <prefix>/include/rocal.
    get_filename_component(ROCAL_LIB_DIR "${ROCAL_LIBRARY}" DIRECTORY)
    get_filename_component(ROCAL_INSTALL_PREFIX "${ROCAL_LIB_DIR}" DIRECTORY)
    set(ROCAL_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ROCAL_INSTALL_PREFIX}/include
        ${ROCAL_INSTALL_PREFIX}/include/rocal
    )
    find_path(MIVisionX_INCLUDE_DIR NAMES vx_ext_amd.h PATHS ${MIVisionX_PATH}/include/mivisionx ${ROCM_PATH}/include/mivisionx NO_DEFAULT_PATH)
    if(MIVisionX_INCLUDE_DIR)
        list(APPEND ROCAL_INCLUDE_DIRS ${MIVisionX_INCLUDE_DIR})
    endif()
endif()

# FFmpeg is mandatory for the plugin.
find_package(FFmpeg REQUIRED)

# The plugin shares rocAL's internal C++ types (e.g. DecoderConfig), whose layout depends on
# ENABLE_HIP, so the plugin must use the backend the selected rocAL was built with.
if(ROCAL_VIDEO_PLUGIN_IN_TREE)
    set(ROCAL_BUILD_CONFIG_HEADER "${CMAKE_BINARY_DIR}/rocAL/rocal_build_config.h")
else()
    set(ROCAL_BUILD_CONFIG_HEADER "${ROCAL_INSTALL_PREFIX}/include/rocal/rocal_build_config.h")
endif()
if(NOT EXISTS "${ROCAL_BUILD_CONFIG_HEADER}")
    message(FATAL_ERROR "-- ${Red}rocal_video_plugin: ${ROCAL_BUILD_CONFIG_HEADER} not found; the selected rocAL does not support the video plugin${ColourReset}")
endif()
file(STRINGS "${ROCAL_BUILD_CONFIG_HEADER}" ROCAL_CONFIG_ENABLE_HIP_LINE REGEX "^#define ROCAL_CONFIG_ENABLE_HIP [01]$")
string(REGEX MATCH "[01]$" ROCAL_CONFIG_ENABLE_HIP "${ROCAL_CONFIG_ENABLE_HIP_LINE}")

set(ROCAL_VIDEO_PLUGIN_HIP OFF)
if(ROCAL_CONFIG_ENABLE_HIP)
    find_package(HIP QUIET)
    if(NOT HIP_FOUND)
        message(FATAL_ERROR "-- ${Red}rocal_video_plugin: rocAL was built with the HIP backend but HIP was not found. Install the HIP development package or set ROCM_PATH/CMAKE_PREFIX_PATH to the ROCm installation${ColourReset}")
    endif()
    find_package(hip QUIET)
    set(ROCAL_VIDEO_PLUGIN_HIP ON)
endif()
message("-- ${White}rocal_video_plugin: rocAL backend from ${ROCAL_BUILD_CONFIG_HEADER} -- ENABLE_HIP=${ROCAL_CONFIG_ENABLE_HIP}${ColourReset}")

# rocDecode is optional and only required on HIP builds.
set(ROCAL_VIDEO_PLUGIN_ROCDECODE OFF)
if(ROCAL_VIDEO_PLUGIN_HIP)
    find_package(rocdecode 1.0.0 QUIET)
    if(rocdecode_FOUND)
        set(ROCAL_VIDEO_PLUGIN_ROCDECODE ON)
    endif()
endif()

include_directories(
    ${ROCAL_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIR}
    ${AVCODEC_INCLUDE_DIR}
    ${SWSCALE_INCLUDE_DIR}
    ${AVFORMAT_INCLUDE_DIR}
)

set(ROCAL_VIDEO_PLUGIN_SOURCES
    source/register.cpp
    source/ffmpeg_video_decoder.cpp
    source/video_properties.cpp
)

if(ROCAL_VIDEO_PLUGIN_HIP AND ROCAL_VIDEO_PLUGIN_ROCDECODE)
    # Derive the rocDecode utility source tree from the package-provided include
    # directory so it works with both flat and version-sliced ROCm installs.
    get_filename_component(rocdecode_PREFIX_DIR "${rocdecode_INCLUDE_DIR}" DIRECTORY)
    set(rocdecode_UTILS_DIR "${rocdecode_PREFIX_DIR}/share/rocdecode/utils")
    set(rocdecode_ROCVIDEODECODE_DIR "${rocdecode_UTILS_DIR}/rocvideodecode")
    # The rocDecode runtime package may not ship the utility source files.
    # If they are missing, disable rocDecode support in the plugin instead of
    # failing configure.
    if(NOT EXISTS "${rocdecode_ROCVIDEODECODE_DIR}/roc_video_dec.cpp"
       OR NOT EXISTS "${rocdecode_UTILS_DIR}/resize_kernels.cpp"
       OR NOT EXISTS "${rocdecode_UTILS_DIR}/colorspace_kernels.cpp")
        message("-- ${Yellow}NOTE: rocDecode utility sources missing at ${rocdecode_UTILS_DIR}; disabling rocDecode video decoder in plugin${ColourReset}")
        set(ROCAL_VIDEO_PLUGIN_ROCDECODE OFF)
    elseif(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message("-- ${Yellow}NOTE: ${CMAKE_CXX_COMPILER} cannot compile the rocDecode HIP kernels; disabling rocDecode video decoder in plugin (use amdclang++)${ColourReset}")
        set(ROCAL_VIDEO_PLUGIN_ROCDECODE OFF)
    else()
        list(APPEND ROCAL_VIDEO_PLUGIN_SOURCES source/rocdec_video_decoder.cpp)
        list(APPEND ROCAL_VIDEO_PLUGIN_SOURCES ${rocdecode_ROCVIDEODECODE_DIR}/roc_video_dec.cpp)
        include_directories(
            ${ROCM_PATH}/include
            ${rocdecode_INCLUDE_DIR}
            ${rocdecode_ROCVIDEODECODE_DIR}
            ${rocdecode_UTILS_DIR}
        )
        # The rocDecode decoder calls these kernels, so they must ship with it rather than rely
        # on optional objects in the prebuilt core.
        set(ROCAL_VIDEO_PLUGIN_KERNEL_SOURCES
            ${rocdecode_UTILS_DIR}/resize_kernels.cpp
            ${rocdecode_UTILS_DIR}/colorspace_kernels.cpp
        )
        set_source_files_properties(${ROCAL_VIDEO_PLUGIN_KERNEL_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
        add_library(rocal_video_plugin_hip OBJECT ${ROCAL_VIDEO_PLUGIN_KERNEL_SOURCES})
        set_target_properties(rocal_video_plugin_hip PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(rocal_video_plugin_hip PRIVATE __HIP_PLATFORM_AMD__)
        target_include_directories(rocal_video_plugin_hip PRIVATE ${rocdecode_UTILS_DIR})
        target_link_libraries(rocal_video_plugin_hip PRIVATE hip::device)
        list(APPEND ROCAL_VIDEO_PLUGIN_SOURCES $<TARGET_OBJECTS:rocal_video_plugin_hip>)
    endif()
endif()

if(ROCAL_VIDEO_PLUGIN_IN_TREE)
    set(ROCAL_VIDEO_ROCDECODE_AVAILABLE ${ROCAL_VIDEO_PLUGIN_ROCDECODE} PARENT_SCOPE)
endif()

if(ROCAL_VIDEO_PLUGIN_STATIC)
    add_library(${PROJECT_NAME} STATIC ${ROCAL_VIDEO_PLUGIN_SOURCES})
else()
    add_library(${PROJECT_NAME} SHARED ${ROCAL_VIDEO_PLUGIN_SOURCES})
endif()

# FFMPEG multi-version support
if(_FFMPEG_AVCODEC_VERSION VERSION_LESS_EQUAL 58.134.100)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_AVCODEC_GREATER_THAN_58_134=0)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_AVCODEC_GREATER_THAN_58_134=1)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE ROCAL_VIDEO)
if(ROCAL_VIDEO_PLUGIN_HIP)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_HIP=1 __HIP_PLATFORM_AMD__)
    if(ROCAL_VIDEO_PLUGIN_ROCDECODE)
        target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_ROCDECODE)
    endif()
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_HIP=0)
endif()

# Export the plugin's own include directory (rocal_video_plugin/register.h) as a usage
# requirement so consumers that link this target can include the registration header,
# e.g. to register a statically-built plugin explicitly.
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${ROCAL_TARGET}
    ${FFMPEG_LIBRARIES}
)

# When HIP is enabled, register.cpp reaches hip/hip_runtime_api.h through DecoderConfig, so
# link hip::host to pull in HIP's include requirements rather than depending on an unrelated
# optional package to supply the ROCm include directory.
if(ROCAL_VIDEO_PLUGIN_HIP)
    find_package(hip QUIET)
    if(TARGET hip::host)
        target_link_libraries(${PROJECT_NAME} PUBLIC hip::host)
    endif()
endif()

if(ROCAL_VIDEO_PLUGIN_HIP AND ROCAL_VIDEO_PLUGIN_ROCDECODE)
    target_link_libraries(${PROJECT_NAME} PUBLIC rocdecode::rocdecode)
endif()

if(NOT DEFINED rocal_VERSION)
    set(rocal_VERSION "2.5.0")
endif()
if(NOT DEFINED rocal_VERSION_MAJOR)
    string(REGEX MATCH "^([0-9]+)" rocal_VERSION_MAJOR "${rocal_VERSION}")
endif()

# Match the core library install layout (lib, not lib64) so the runtime loader's library
# search paths line up with where the plugin is installed. Honor an explicit user override.
if(NOT DEFINED CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR "lib" CACHE PATH "Library install directory")
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
# ASAN packages select only the asan component; ship the plugin there too so an ASAN package
# in plugin mode still contains the video implementation alongside the core library.
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan NAMELINK_SKIP
)

# Install plugin sources under COMPONENT dev (issue comment #2).
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/rocal/plugins/video
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.cpp"
    PATTERN "*.hpp"
    PATTERN "CMakeLists.txt"
)
