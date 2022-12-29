# Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.0)
project(rocal_pybind)

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)
set(ROCM_PATH /opt/rocm CACHE PATH "Deafult ROCm Installation Path")

set(CMAKE_CXX_STANDARD 17)
set(BUILD_ROCAL_PYBIND true)

find_package(AMDRPP QUIET)
find_package(MIVisionX QUIET)
find_package(TurboJpeg QUIET)

# Backend
if(NOT DEFINED BACKEND)
    set(BACKEND "HIP")   # set default backend to HIP
else()
    if("${BACKEND}" STREQUAL "OPENCL" OR "${BACKEND}" STREQUAL "OCL" OR "${BACKEND}" STREQUAL "OpenCL")
        set(BACKEND "OPENCL")
    elseif("${BACKEND}" STREQUAL "HIP" OR "${BACKEND}" STREQUAL "hip")
        set(BACKEND "HIP")
    elseif("${BACKEND}" STREQUAL "host" OR "${BACKEND}" STREQUAL "CPU" OR "${BACKEND}" STREQUAL "cpu")
        set(BACKEND "CPU")
        set(GPU_SUPPORT OFF)
    else()
        message("-- ${Red}Warning: rocAL PyBind backend option: ${BACKEND} -- unknown${ColourReset}")
        message("-- ${Red}Warning: rocAL PyBind default backend will be enforced${ColourReset}")
        set(BACKEND "HIP")
    endif()
endif()
message("-- ${BoldBlue}rocAL PyBind Backend set to -- ${BACKEND}${ColourReset}")

# Find HIP Support
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
        message("-- ${White}rocAL PyBind -- Using HIP -- Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER} ${ColourReset})
    else()
        set(BUILD_ROCAL_PYBIND false)
        message("-- ${Yellow}rocAL PyBind requires HIP, Not Found${ColourReset}")
    endif()
endif()

# Find OpenCL Support
if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message("-- ${White}rocAL PyBind -- Using OpenCL -- Path:" ${OpenCL_LIBRARIES} "\tVersion:" ${CL_TARGET_OpenCL_VERSION})
    else()
        set(BUILD_ROCAL_PYBIND false)
        message("-- ${Yellow}rocAL PyBind requires OpenCL, Not Found${ColourReset}")
    endif()
endif()

# Dependency Check
# Find AMD RPP Support
if(NOT AMDRPP_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL PyBind requires AMD RPP, Not Found${ColourReset}")
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
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Red}ERROR: rocAL PyBind -- RPP with OpenCL backend Not Found${ColourReset}")
        endif()
    elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        if (NOT RPP_BACKEND_HIP_FOUND)
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Red}ERROR: rocAL PyBind -- RPP with the HIP backend Not Found${ColourReset}")
        endif()
    endif()
endif()
# MIVisionX
if(NOT MIVisionX_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL PyBind library requires AMD's MIVisionX library, Not Found ${ColourReset}")
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
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Red}ERROR: rocAL PyBind -- MIVisionX with OpenCL backend Not Found${ColourReset}")
        endif()
    elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        if (NOT MIVISIONX_BACKEND_HIP_FOUND)
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Red}ERROR: rocAL PyBind -- MIVisionX with the HIP backend Not Found${ColourReset}")
        endif()
    endif()
endif()

# Turbo JPEG
if(NOT TurboJpeg_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL PyBind requires TurboJpeg, Not Found${ColourReset}")
endif()

if(${BUILD_ROCAL_PYBIND})

    # AMD RPP
    include_directories(${AMDRPP_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${AMDRPP_LIBRARIES})
    # MIVisionX
    include_directories(${MIVisionX_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OPENVX_LIBRARIES} ${VXRPP_LIBRARIES})
    # TurboJPEG
    include_directories(${TurboJpeg_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${TurboJpeg_LIBRARIES})
    # rocAL
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocal)

    link_directories(${ROCM_PATH}/lib)

    include_directories(../rocAL/include/api/
                        third_party_lib/
                        ${MIVisionX_INCLUDE_DIRS}
                    )

    add_subdirectory(third_party_lib/pybind11)

    pybind11_add_module(rocal_pybind rocal_pybind.cpp)

    target_link_libraries(rocal_pybind PRIVATE ${LINK_LIBRARY_LIST})
    message("-- ${White}rocal_pybind -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")
    message("-- ${White}rocal_pybind -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")

    # install rocAL pybind libs -- {ROCM_PATH)/lib
    install(TARGETS rocal_pybind DESTINATION ${CMAKE_INSTALL_LIBDIR})
    message("-- ${Green}Radeon Augmentation Library Python Binding - rocal_pybind module added ${ColourReset}")
else()
    message("-- ${Red}WARNING: rocAL PyBind Module excluded - Dependency Failure${ColourReset}")
endif()
