################################################################################
#
# MIT License
#
# Copyright (c) 2018 - 2026 Advanced Micro Devices, Inc.
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

project (performance_tests_with_depth)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)

find_package(AMDRPP QUIET)

# Take the rocAL headers and library from one prefix, the rocAL this test belongs to,
# so they always describe the same build; see cmake/RocalTestRoot.cmake in the test suite.
# The helper comes from the suite this test sits in. A test directory copied out of
# <prefix>/share/rocal/test has no suite next to it, so it uses the one installed under ROCM_PATH.
# Without either, as with a rocAL install that predates the helper, link rocal from the library
# search path with the ROCM_PATH headers.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/RocalTestRoot.cmake OPTIONAL RESULT_VARIABLE ROCAL_TEST_ROOT_MODULE)
if(NOT ROCAL_TEST_ROOT_MODULE)
    include(${ROCM_PATH}/share/rocal/test/cmake/RocalTestRoot.cmake OPTIONAL RESULT_VARIABLE ROCAL_TEST_ROOT_MODULE)
endif()
if(ROCAL_TEST_ROOT_MODULE)
    rocal_test_find_rocal()
else()
    set(ROCAL_LIBRARY rocal)
    set(ROCAL_INCLUDE_DIR ${ROCM_PATH}/include/rocal)
endif()
include_directories(${ROCM_PATH}/include ${ROCAL_INCLUDE_DIR})
link_directories(${ROCM_PATH}/lib)

file(GLOB My_Source_Files ./*.cpp)
add_executable(${PROJECT_NAME} ${My_Source_Files})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake ${ROCM_PATH}/share/rocal/test/cmake)
find_package(OpenCVComponents MODULE QUIET)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall ")
# librocal carries Python C-API symbols (PythonFunction operator); link libpython to resolve them
find_package(Python3 QUIET COMPONENTS Development.Embed)
target_link_libraries(${PROJECT_NAME} ${ROCAL_LIBRARY} $<TARGET_NAME_IF_EXISTS:Python3::Python>)
