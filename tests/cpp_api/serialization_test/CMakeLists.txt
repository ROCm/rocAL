################################################################################
#
# MIT License
#
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc.
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
cmake_minimum_required(VERSION 3.10)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
elseif(NOT DEFINED CMAKE_CXX_COMPILER AND NOT EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
endif()

project(serialization_test)

# Take the rocAL headers and library from one prefix, the rocAL this test belongs to,
# so they always describe the same build; see ../cmake/RocalTestRoot.cmake.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/RocalTestRoot.cmake)
rocal_test_find_rocal(${CMAKE_CURRENT_SOURCE_DIR}/..)

if(ROCAL_FOUND)
    message("-- ${White}${PROJECT_NAME}: Using rocAL -- \n\tLibraries:${ROCAL_LIBRARY} \n\tIncludes:${ROCAL_INCLUDE_DIR}${ColourReset}")
endif()

if(NOT ROCAL_FOUND)
    message("-- ${Yellow}${PROJECT_NAME} requires rocAL. Install rocAL before running CTests")
else()
    set(RPP_BACKEND_HIP_FOUND 0)
    if(EXISTS ${ROCM_PATH}/include/rpp/rpp_backend.h)
        file(READ ${ROCM_PATH}/include/rpp/rpp_backend.h RPP_BACKEND_FILE)
        string(REGEX MATCH "RPP_BACKEND_HIP ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    endif()
    if(NOT DEFINED BACKEND AND RPP_BACKEND_HIP_FOUND)
        set(BACKEND "HIP")
    elseif(NOT DEFINED BACKEND)
        set(BACKEND "CPU")
    endif()
endif(NOT ROCAL_FOUND)
message("-- ${BoldBlue}Backend set to -- ${BACKEND}${ColourReset}")

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rocal/test/cmake)

# Find required packages
find_package(AMDRPP QUIET)

include_directories(${ROCM_PATH}/include ${ROCAL_INCLUDE_DIR})
link_directories(${ROCM_PATH}/lib)

# Source files
file(GLOB Source_Files ./*.cpp)
add_executable(${PROJECT_NAME} ${Source_Files})

# OpenCV configuration
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake ${ROCM_PATH}/share/rocal/test/cmake)
find_package(OpenCVComponents MODULE QUIET)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Link libraries
# librocal carries Python C-API symbols (PythonFunction operator); link libpython to resolve them
find_package(Python3 QUIET COMPONENTS Development.Embed)
target_link_libraries(${PROJECT_NAME} ${ROCAL_LIBRARY} $<TARGET_NAME_IF_EXISTS:Python3::Python>)
