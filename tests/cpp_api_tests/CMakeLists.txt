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

add_test(
  NAME
    rocAL_basic_test
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/rocAL_basic_test"
                              "${CMAKE_CURRENT_BINARY_DIR}/rocAL_basic_test"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "rocal_basic_test"
            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet-val.txt 0 1 224 224 0
)

add_test(
  NAME
  rocAL_dataloader
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/rocAL_dataloader"
                              "${CMAKE_CURRENT_BINARY_DIR}/rocAL_dataloader"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "rocal_dataloader"
            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet
)

add_test(
  NAME
  rocAL_dataloader_mt
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/rocAL_dataloader_mt"
                              "${CMAKE_CURRENT_BINARY_DIR}/rocAL_dataloader_mt"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "rocal_dataloader_mt"
            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet
)

add_test(
  NAME
  rocAL_dataloader_tf
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/rocAL_dataloader_tf"
                              "${CMAKE_CURRENT_BINARY_DIR}/rocAL_dataloader_tf"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "rocal_dataloader_tf"
            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet
)

add_test(
  NAME
    rocAL_performace_tests
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/rocAL_performace_tests"
                              "${CMAKE_CURRENT_BINARY_DIR}/rocAL_performace_tests"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "rocal_performance_tests"
            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet
)

add_test(
  NAME
  rocAL_performace_tests_with_depth
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/rocAL_performace_tests_with_depth"
                              "${CMAKE_CURRENT_BINARY_DIR}/rocAL_performace_tests_with_depth"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "rocal_performance_tests_with_depth"
            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet
)

add_test(
  NAME
    rocAL_unittests
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/rocAL_unittests"
                              "${CMAKE_CURRENT_BINARY_DIR}/rocAL_unittests"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "rocal_unittests"
            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet
)

add_test(
  NAME
  rocAL_video_unittests
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/rocAL_video_unittests"
                              "${CMAKE_CURRENT_BINARY_DIR}/rocAL_video_unittests"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "rocal_video_unittests"
            ${CMAKE_SOURCE_DIR}/data/videos/AMD_driving_virtual_20.mp4
)
