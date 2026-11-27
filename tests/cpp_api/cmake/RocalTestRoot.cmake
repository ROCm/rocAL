################################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

# rocal_test_find_rocal()
#
# Sets ROCAL_FOUND, ROCAL_ROOT, ROCAL_LIBRARY and ROCAL_INCLUDE_DIR to the rocAL a C++ test builds against,
# taking the library and the headers from the same prefix. The test suite is the directory above
# this module (tests/cpp_api, installed as <prefix>/share/rocal/test). It is located from this
# file rather than from the test, so a test directory copied out of the installed suite that
# loads this module from <prefix>/share/rocal/test/cmake still resolves <prefix>. The first
# candidate that holds both librocal and include/rocal/rocal_api.h is used:
#   1. ROCAL_ROOT, when set. The test suite passes it to every test project.
#   2. <prefix>, when the suite is <prefix>/<datadir>/rocal/test. This is the rocAL the tests
#      were installed with. It comes before CMAKE_INSTALL_PREFIX because a standalone configure
#      defaults that to /usr/local, which may hold an unrelated rocAL.
#   3. CMAKE_INSTALL_PREFIX.
#   4. ROCM_PATH.
# lib, lib64 and lib/<multiarch> are checked. ROCAL_LIBRARY is a full path, so the linker cannot
# resolve -lrocal to a librocal from another directory on its search path, and CMake adds that
# directory to the build RPATH so the test also loads the same library at run time.
# An explicit ROCAL_ROOT without rocAL is an error. When no candidate holds rocAL, librocal from
# the library search path is used with the ROCM_PATH/include/rocal headers; if either is missing,
# ROCAL_FOUND is FALSE and ROCAL_LIBRARY is the bare name "rocal".
get_filename_component(_ROCAL_TEST_SUITE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

function(rocal_test_find_rocal)
    set(_candidates ${ROCAL_ROOT})
    if(_ROCAL_TEST_SUITE_DIR MATCHES "^(.*)/rocal/test$")
        # <datadir> may be more than one level deep (e.g. share/rocm), so walk up to the prefix.
        set(_dir "${CMAKE_MATCH_1}")
        foreach(_level RANGE 2)
            get_filename_component(_dir "${_dir}" DIRECTORY)
            if(EXISTS "${_dir}/include/rocal/rocal_api.h")
                list(APPEND _candidates "${_dir}")
                break()
            endif()
        endforeach()
    endif()
    list(APPEND _candidates ${CMAKE_INSTALL_PREFIX} ${ROCM_PATH})

    set(_libdirs lib lib64)
    if(CMAKE_LIBRARY_ARCHITECTURE)
        list(APPEND _libdirs lib/${CMAKE_LIBRARY_ARCHITECTURE})
    endif()
    set(_library_name "${CMAKE_SHARED_LIBRARY_PREFIX}rocal${CMAKE_SHARED_LIBRARY_SUFFIX}")
    foreach(_root IN LISTS _candidates)
        if(NOT EXISTS "${_root}/include/rocal/rocal_api.h")
            if(ROCAL_ROOT AND _root STREQUAL ROCAL_ROOT)
                message(FATAL_ERROR "-- ${Red}${PROJECT_NAME}: ROCAL_ROOT=${ROCAL_ROOT} has no include/rocal/rocal_api.h${ColourReset}")
            endif()
            continue()
        endif()
        foreach(_libdir IN LISTS _libdirs)
            if(EXISTS "${_root}/${_libdir}/${_library_name}")
                message("-- ${White}${PROJECT_NAME}: using rocAL from ${_root}${ColourReset}")
                set(ROCAL_FOUND TRUE PARENT_SCOPE)
                set(ROCAL_ROOT "${_root}" PARENT_SCOPE)
                set(ROCAL_LIBRARY "${_root}/${_libdir}/${_library_name}" PARENT_SCOPE)
                set(ROCAL_INCLUDE_DIR "${_root}/include/rocal" PARENT_SCOPE)
                return()
            endif()
        endforeach()
        if(ROCAL_ROOT AND _root STREQUAL ROCAL_ROOT)
            string(REPLACE ";" ", " _libdir_list "${_libdirs}")
            message(FATAL_ERROR "-- ${Red}${PROJECT_NAME}: ROCAL_ROOT=${ROCAL_ROOT} has no ${_library_name} in ${_libdir_list}${ColourReset}")
        endif()
    endforeach()

    set(ROCAL_INCLUDE_DIR "${ROCM_PATH}/include/rocal" PARENT_SCOPE)
    unset(ROCAL_SEARCH_PATH_LIBRARY CACHE)
    find_library(ROCAL_SEARCH_PATH_LIBRARY NAMES rocal HINTS ${ROCM_PATH}/lib)
    if(ROCAL_SEARCH_PATH_LIBRARY AND EXISTS "${ROCM_PATH}/include/rocal/rocal_api.h")
        message("-- ${Yellow}${PROJECT_NAME}: rocAL not found under ${_candidates}; using ${ROCAL_SEARCH_PATH_LIBRARY} with headers from ${ROCM_PATH}/include/rocal${ColourReset}")
        set(ROCAL_FOUND TRUE PARENT_SCOPE)
        set(ROCAL_LIBRARY "${ROCAL_SEARCH_PATH_LIBRARY}" PARENT_SCOPE)
        return()
    endif()
    message("-- ${Yellow}rocAL not found under ${_candidates}; linking rocal from the library search path${ColourReset}")
    set(ROCAL_FOUND FALSE PARENT_SCOPE)
    set(ROCAL_LIBRARY rocal PARENT_SCOPE)
endfunction()
