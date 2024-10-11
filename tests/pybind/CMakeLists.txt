################################################################################
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
cmake_minimum_required (VERSION 3.5)

project(rocal-pybind-test)

# make test with CTest
enable_testing()
include(CTest)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- ${PROJECT_NAME} INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# find rocAL
find_library(ROCAL_LIBRARY NAMES rocal HINTS ${ROCM_PATH}/lib)
find_library(ROCAL_PYBIND_LIBRARY NAMES rocal_pybind HINTS ${ROCM_PATH}/lib)
find_path(ROCAL_INCLUDE_DIR NAMES rocal_api.h PATHS ${ROCM_PATH}/include/rocal)

set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 QUIET)

# OpenVX Tests
if(Python3_FOUND)
  message("-- ${PROJECT_NAME} INFO:ROCAL_PYBIND_LIBRARY -- ${ROCAL_PYBIND_LIBRARY}")
  # 1 - decoders test
  add_test(NAME rocal_pybind_test_decoders
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/decoders_test.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  # 2 - functions test
  add_test(NAME rocal_pybind_test_functions
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/functions_test.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  # 3 - pipelines test
  add_test(NAME rocal_pybind_test_pipelines
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pipelines_test.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  # 4 - randoms test
  add_test(NAME rocal_pybind_test_randoms
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/randoms_test.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  # 5 - readers test
  add_test(NAME rocal_pybind_test_readers
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/readers_test.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  # 6 - types test
  add_test(NAME rocal_pybind_test_types
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/types_test.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif(Python3_FOUND)
