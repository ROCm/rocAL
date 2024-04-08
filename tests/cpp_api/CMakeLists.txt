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

# basic_test
add_test(
  NAME
    basic_test_cpu
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/basic_test"
                              "${CMAKE_CURRENT_BINARY_DIR}/basic_test"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "basic_test"
            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet-val.txt 1 0 224 224
)
add_test(NAME basic_test_gpu 
              COMMAND basic_test 
              ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet-val.txt 1 0 224 224
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/basic_test)
add_test(NAME basic_test_gray 
              COMMAND basic_test 
              ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet-val.txt 1 1 224 224 0
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/basic_test)
add_test(NAME basic_test_rgb 
              COMMAND basic_test 
              ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet-val.txt 1 1 224 224 1
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/basic_test)

# TBD - dataloader unit test options non-functional - NEEDS TO BE ADDED ONCE RESOLVED
#add_test(
#  NAME
#  dataloader
#  COMMAND
#    "${CMAKE_CTEST_COMMAND}"
#            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/dataloader"
#                              "${CMAKE_CURRENT_BINARY_DIR}/dataloader"
#            --build-generator "${CMAKE_GENERATOR}"
#            --test-command "dataloader"
#            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet
#)

# dataloader_multithread
add_test(
  NAME
  rocAL_dataloader_mt_cpu
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/dataloader_multithread"
                              "${CMAKE_CURRENT_BINARY_DIR}/dataloader_multithread"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "dataloader_multithread"
            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet 0
)
add_test(NAME rocAL_dataloader_mt_gpu 
              COMMAND dataloader_multithread 
              ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet 1
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dataloader_multithread)

# TBD - dataloader_tf unit test non-functional
#add_test(
#  NAME
#  dataloader_tf
#  COMMAND
#    "${CMAKE_CTEST_COMMAND}"
#            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/dataloader_tf"
#                              "${CMAKE_CURRENT_BINARY_DIR}/dataloader_tf"
#            --build-generator "${CMAKE_GENERATOR}"
#            --test-command "dataloader_tf"
#            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet
#)

# performance_tests
# TBD - peformance test needs to run with default options
add_test(
  NAME
    rocAL_performance_tests_cpu
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/performance_tests"
                              "${CMAKE_CURRENT_BINARY_DIR}/performance_tests"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "performance_tests"
            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet 224 224 1 16 0
)
add_test(NAME rocAL_performance_tests_gpu 
              COMMAND performance_tests 
              ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet 224 224 1 16 1
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/performance_tests)

# performance_tests_with_depth
add_test(
  NAME
  performance_tests_with_depth_cpu
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/performance_tests_with_depth"
                              "${CMAKE_CURRENT_BINARY_DIR}/performance_tests_with_depth"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "performance_tests_with_depth"
            ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet 224 224 1 1 1 0
)
add_test(NAME performance_tests_with_depth_gpu 
              COMMAND performance_tests_with_depth 
              ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet 224 224 1 1 1 1
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/performance_tests_with_depth)

# unit_tests
add_test(
  NAME
    unit_tests_cpu
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/unit_tests"
                              "${CMAKE_CURRENT_BINARY_DIR}/unit_tests"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "unit_tests"
            0 ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet test 224 224 1 0 1
)
add_test(NAME unit_tests_gpu 
              COMMAND unit_tests 
              0 ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet test 224 224 1 1 1
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/unit_tests)
add_test(NAME unit_tests_gray 
              COMMAND unit_tests 
              0 ${CMAKE_SOURCE_DIR}/data/images/AMD-tinyDataSet test 224 224 1 1 0
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/unit_tests)

# video_tests
add_test(
  NAME
  video_tests
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/video_tests"
                              "${CMAKE_CURRENT_BINARY_DIR}/video_tests"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "video_tests"
            ${CMAKE_SOURCE_DIR}/data/videos/AMD_driving_virtual_20.mp4
)
