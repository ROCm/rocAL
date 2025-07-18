################################################################################
#
# MIT License
#
# Copyright (c) 2018 - 2025 Advanced Micro Devices, Inc.
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
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
elseif(NOT DEFINED CMAKE_CXX_COMPILER AND NOT EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
endif()

project (unit_tests)

find_library(ROCAL_LIBRARY NAMES rocal HINTS ${ROCM_PATH}/lib)
find_path(ROCAL_INCLUDE_DIR NAMES rocal_api.h PATHS ${ROCM_PATH}/include/rocal)

if(ROCAL_LIBRARY AND ROCAL_INCLUDE_DIR)
    set(ROCAL_FOUND TRUE)
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

list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rocal/test/cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

find_package(OpenCV QUIET)
find_package(AMDRPP QUIET)

include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rocal)
link_directories(${ROCM_PATH}/lib)
file(GLOB My_Source_Files ./*.cpp)
add_executable(${PROJECT_NAME} ${My_Source_Files})

if("${BACKEND}" STREQUAL "HIP")
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message("-- ${White}${PROJECT_NAME} -- Using HIP - Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER}${ColourReset})
        include_directories(${HIP_INCLUDE_DIRS} ${HIP_INCLUDE_DIRS}/hip)
        target_link_libraries(${PROJECT_NAME} ${HIP_LIBRARIES})
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=1)
    else()
        set(BACKEND "CPU")
        message("-- ${Yellow}HIP not found! ${PROJECT_NAME} set to build with CPU backend${ColourReset}")
    endif()
endif()

if(OpenCV_FOUND)
    if(${OpenCV_VERSION_MAJOR} EQUAL 3 OR ${OpenCV_VERSION_MAJOR} EQUAL 4)
        message("-- OpenCV Found -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Supported")
        include_directories(${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
        if(${OpenCV_VERSION_MAJOR} EQUAL 4)
            target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=1)
        else()
            target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=0)
        endif()
    else()
        message(FATAL_ERROR "OpenCV Found -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Not Supported")
    endif()
else()
    message(FATAL_ERROR "OpenCV Not Found -- No Display Support")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mf16c -Wall ")
target_link_libraries(${PROJECT_NAME} rocal)
