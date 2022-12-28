# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.0)
project(rocal)

set(BUILD_ROCAL true)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

find_package(AMDRPP QUIET)
find_package(MIVisionX QUIET)
find_package(TurboJpeg QUIET)
find_package(Protobuf QUIET)
find_package(FFmpeg QUIET)
find_package(OpenCV QUIET)
find_package(OpenMP QUIET)
set(BOOST_COMPONENTS filesystem system)
find_package(Boost COMPONENTS ${BOOST_COMPONENTS} QUIET)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads QUIET)

# HIP Backend
if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})
    find_package(HIP QUIET)
    if(HIP_FOUND)
        SET(BUILD_ROCAL true)
        message("-- ${White}rocAL -- Using HIP -- Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER} ${ColourReset})
    else()
        set(BACKEND "CPU")
        message("-- ${Yellow}NOTE: rocAL library requires HIP for BACKEND=HIP, Not Found ${ColourReset}")
        message("-- ${Yellow}NOTE: rocAL Backend set to CPU ${ColourReset}")
    endif()
endif()

#OpenCL Backend
if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(BUILD_ROCAL true)
        message("-- ${White}rocAL -- Using OpenCL -- Path:" ${OpenCL_LIBRARIES} "\tVersion:" ${CL_TARGET_OpenCL_VERSION})
    else()
      set(BACKEND "CPU")
      message("-- ${Yellow}NOTE: rocAL library requires OpenCL for BACKEND=OpenCL, Not Found ${ColourReset}")
      message("-- ${Yellow}NOTE: rocAL Backend set to CPU ${ColourReset}")
  endif()
endif()
# Dependency Check
# AMD RPP
if(NOT AMDRPP_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires AMD's rpp library, Not Found ${ColourReset}")
else()
    #find the RPP backend type
    set(RPP_BACKEND_OPENCL_FOUND 0)
    set(RPP_BACKEND_HIP_FOUND 0)
    if(EXISTS ${ROCM_PATH}/include/rpp/rpp_backend.h)
        file(READ ${ROCM_PATH}/include/rpp/rpp_backend.h RPP_BACKEND_FILE)
        string(REGEX MATCH "RPP_BACKEND_OPENCL ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
        string(REGEX MATCH "RPP_BACKEND_HIP ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    else()
        message(FATAL_ERROR "${Red}ERROR: ${ROCM_PATH}/include/rpp/rpp_backend.h file Not Found. AMD RPP Package REQUIRED ${ColourReset}")
    endif()

    if ("${BACKEND}" STREQUAL "OPENCL" AND OpenCL_FOUND)
        if (NOT RPP_BACKEND_OPENCL_FOUND)
            set(BACKEND "CPU")
            message("-- ${Red}WARNING: RPP with OpenCL backend Not Found (HINT: run setup script with the OCL backend)${ColourReset}")
            message("-- ${Yellow}NOTE: rocAL Backend set to CPU ${ColourReset}")
        endif()
    elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        if (NOT RPP_BACKEND_HIP_FOUND)
            set(BACKEND "CPU")
            message("-- ${Red}WARNING: RPP with the HIP backend Not Found (HINT: run setup script with the HIP backend)${ColourReset}")
            message("-- ${Yellow}NOTE: rocAL Backend set to CPU ${ColourReset}")
        endif()
    endif()
endif()
# MIVisionX
if(NOT MIVisionX_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires AMD's MIVisionX library, Not Found ${ColourReset}")
else()
    #find the MIVisionX backend type
    set(MIVISIONX_BACKEND_OPENCL_FOUND 0)
    set(MIVISIONX_BACKEND_HIP_FOUND 0)
    if(EXISTS ${ROCM_PATH}/include/mivisionx/openvx_backend.h)
        file(READ ${ROCM_PATH}/include/mivisionx/openvx_backend.h MIVISIONX_BACKEND_FILE)
        string(REGEX MATCH "ENABLE_OPENCL ([0-9]*)" _ ${MIVISIONX_BACKEND_FILE})
        set(MIVISIONX_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
        string(REGEX MATCH "ENABLE_HIP ([0-9]*)" _ ${MIVISIONX_BACKEND_FILE})
        set(MIVISIONX_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    else()
        message(FATAL_ERROR "${Red}ERROR: ${ROCM_PATH}/include/mivisionx/openvx_backend.h file Not Found. MIVisionX Package REQUIRED ${ColourReset}")
    endif()

    if ("${BACKEND}" STREQUAL "OPENCL" AND OpenCL_FOUND)
        if (NOT MIVISIONX_BACKEND_OPENCL_FOUND)
            set(BACKEND "CPU")
            message("-- ${Red}WARNING: MIVisionX with OpenCL backend Not Found (HINT: run setup script with the OCL backend)${ColourReset}")
            message("-- ${Yellow}NOTE: rocAL Backend set to CPU ${ColourReset}")
        endif()
    elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        if (NOT MIVISIONX_BACKEND_HIP_FOUND)
            set(BACKEND "CPU")
            message("-- ${Red}WARNING: MIVisionX with the HIP backend Not Found (HINT: run setup script with the HIP backend)${ColourReset}")
            message("-- ${Yellow}NOTE: rocAL Backend set to CPU ${ColourReset}")
        endif()
    endif()
endif()
# Turbo JPEG
if(NOT TurboJpeg_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires TurboJpeg, Not Found ${ColourReset}")
endif()
# Google ProtoBuff
if(NOT Protobuf_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires Protobuf, Not Found ${ColourReset}")
endif()
# OpenMP
if(NOT OpenMP_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires OpenMP, Not Found ${ColourReset}")
endif()
# BOOST
if(NOT Boost_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires Boost, Not Found ${ColourReset}")
endif()
# Threads
if(NOT Threads_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires Threads, Not Found ${ColourReset}")
endif()

if(${BUILD_ROCAL})
    # AMD RPP
    include_directories(${AMDRPP_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${AMDRPP_LIBRARIES})
    # MIVisionX
    include_directories(${MIVisionX_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OPENVX_LIBRARIES} ${VXRPP_LIBRARIES})
    # OpenMP
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} OpenMP::OpenMP_CXX)
    # Threads
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} Threads::Threads)
    # BOOST
    include_directories(${Boost_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})
    # TurboJPEG
    include_directories(${TurboJpeg_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${TurboJpeg_LIBRARIES})
    # Protobuf
    include_directories(${PROTOBUF_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${PROTOBUF_LIBRARIES})
    # LMDB -- TBD: FIND PACKAGE Required
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} lmdb)

    # rocAL uses C++ 17 features
    set(CMAKE_CXX_STANDARD 17)
    message("-- ${Green}ROCm Augmentation Library -- rocAL added${ColourReset}")

    include_directories(
                ${CMAKE_CURRENT_BINARY_DIR}
                ./include/api/
                ./include/augmentations/
                ./include/augmentations/color_augmentations/
                ./include/augmentations/effects_augmentations/
                ./include/augmentations/geometry_augmentations/
                ./include/decoders/image/
                ./include/decoders/video/
                ./include/device/
                ./include/loaders/
                ./include/loaders/image/
                ./include/loaders/video/
                ./include/meta_data/
                ./include/parameters/
                ./include/pipeline/
                ./include/readers/image/
                ./include/readers/video/
                ./../third_party/rapidjson/include/
                )

    link_directories(${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR})

    if("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        link_directories(${HIP_PATH}/${CMAKE_INSTALL_LIBDIR})
        include_directories(${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR} rocAL_hip)
        add_subdirectory(rocAL_hip)
    endif()

    protobuf_generate_cpp(TF_PROTO_SRCS TF_PROTO_HEADERS proto/example.proto proto/feature.proto)
    protobuf_generate_cpp(CAFFE2_PROTO_SRCS CAFFE2_PROTO_HEADERS proto/caffe2_protos.proto)
    protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HEADERS proto/caffe_protos.proto)
    link_directories(${AMDRPP_LIBRARIES_DIRS} ${TurboJpeg_LIBRARIES_DIRS} ${PROTOBUF_LIBRARY_DIRS} /usr/local/lib/)

    file(GLOB_RECURSE SOURCES "./source/*.cpp")
    add_library(${PROJECT_NAME} SHARED ${SOURCES} ${TF_PROTO_SRCS} ${TF_PROTO_HEADERS} ${CAFFE_PROTO_HEADERS} ${CAFFE_PROTO_SRCS} ${CAFFE2_PROTO_SRCS} ${CAFFE2_PROTO_HEADERS})

    if("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        add_dependencies(${PROJECT_NAME} rocAL_hip)
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} $<TARGET_OBJECTS:rocAL_hip>)
        target_compile_definitions(${PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=1)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=0)
        message("-- ${White}rocAL built with HIP Backend${ColourReset}")
    elseif("${BACKEND}" STREQUAL "OPENCL" AND OPENCL_FOUND)
        include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=0)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=1)
        message("-- ${White}rocAL built with OpenCL Backend${ColourReset}")
    else()
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=0)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=0)
        message("-- ${White}rocAL built with CPU only Backend${ColourReset}")
    endif()

    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_SIMD=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DBG_TIMING=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DBGINFO=0)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DBGLOG=0 WRNLOG=0)
    target_compile_definitions(${PROJECT_NAME} PUBLIC TFRECORD_ENABLE=0)
    message("-- ${White}rocAL built with ENABLE_SIMD")
    #OpenCV
    if (OpenCV_FOUND)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=1)
        include_directories(${OpenCV_INCLUDE_DIRS})
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OpenCV_LIBRARIES})
        if(${OpenCV_VERSION_MAJOR} EQUAL 4)
            target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=1)
        else()
	        target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=0)
        endif()
    else()
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=0)
    endif()
    # -Wall -- Enable most warning messages
    # -mavx2 -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and AVX2 built-in functions and code generation
    # -mfma -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and FMA built-in functions and code generation
    # -mf16c -- Support F16C built-in functions and code generation
    # -Wno-deprecated-declarations -- Do not warn about uses of functions, variables, and types marked as deprecated by using the deprecated attribute
    # -std=gnu++17 -- Conform to the ISO 2017 C++ standard with GNU extensions
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mavx2 -mfma -mf16c -Wno-deprecated-declarations -std=gnu++17")
    message("-- ${White}rocAL -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")

    if(NOT FFMPEG_FOUND)
        message("-- ${Yellow}NOTE: rocAL library is going to be built without video decode functionality ${ColourReset}")
    else()
        include_directories(${AVUTIL_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR} ${SWSCALE_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR})
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${FFMPEG_LIBRARIES})
        message("-- ${White}rocAL library is going to be built with video decode functionality ${ColourReset}")
        target_compile_definitions(${PROJECT_NAME} PUBLIC -DROCAL_VIDEO)
    endif()
    target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})
    message("-- ${White}rocAL -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")

    # install rocAL libs -- {ROCM_PATH)/lib
    install(TARGETS rocal DESTINATION ${CMAKE_INSTALL_LIBDIR})
    if(BUILD_DEV)
        # install rocAL include files -- {ROCM_PATH}/include/rocAL/
        install(FILES
                include/api/rocal_api.h
                include/api/rocal_api_info.h
                include/api/rocal_api_augmentation.h
                include/api/rocal_api_data_loaders.h
                include/api/rocal_api_types.h
                include/api/rocal_api_data_transfer.h
                include/api/rocal_api_parameters.h
                include/api/rocal_api_meta_data.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocal
        )
    endif(BUILD_DEV)
else()
    message(FATAL_ERROR "-- ${Red}rocAL dependencies not satisfied${ColourReset}")
endif()
