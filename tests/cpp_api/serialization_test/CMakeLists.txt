################################################################################
#
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

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

project(serialization_test)

# Find required packages
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rocal)

# Link directories
link_directories(${ROCM_PATH}/lib)

# Source files
file(GLOB My_Source_Files ./*.cpp)
add_executable(${PROJECT_NAME} ${My_Source_Files})

# OpenCV configuration
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

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Link libraries
target_link_libraries(${PROJECT_NAME} rocal)
