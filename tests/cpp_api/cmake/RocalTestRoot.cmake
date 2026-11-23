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

# rocal_test_find_rocal(<suite_dir>)
#
# Sets ROCAL_FOUND, ROCAL_ROOT, ROCAL_LIBRARY and ROCAL_INCLUDE_DIR to the rocAL a C++ test builds against,
# taking the library and the headers from the same prefix. <suite_dir> is the directory of the
# test suite (tests/cpp_api, installed as <prefix>/share/rocal/test). The first candidate that
# holds both librocal and include/rocal/rocal_api.h is used:
#   1. ROCAL_ROOT, when set. The test suite passes it to every test project.
#   2. <prefix>, when <suite_dir> is <prefix>/share/rocal/test. This is the rocAL the tests were
#      installed with. It comes before CMAKE_INSTALL_PREFIX because a standalone configure
#      defaults that to /usr/local, which may hold an unrelated rocAL.
#   3. CMAKE_INSTALL_PREFIX.
#   4. ROCM_PATH.
# Both lib and lib64 are checked. ROCAL_LIBRARY is a full path, so the linker cannot resolve
# -lrocal to a librocal from another directory on its search path, and CMake adds that
# directory to the build RPATH so the test also loads the same library at run time.
# When no candidate holds rocAL, ROCAL_LIBRARY is the bare name "rocal" for the linker search
# path and ROCAL_INCLUDE_DIR is ROCM_PATH/include/rocal.
function(rocal_test_find_rocal suite_dir)
    set(_candidates ${ROCAL_ROOT})
    get_filename_component(_suite_dir "${suite_dir}" ABSOLUTE)
    if(_suite_dir MATCHES "^(.*)/share/rocal/test$")
        list(APPEND _candidates "${CMAKE_MATCH_1}")
    endif()
    list(APPEND _candidates ${CMAKE_INSTALL_PREFIX} ${ROCM_PATH})

    set(_library_name "${CMAKE_SHARED_LIBRARY_PREFIX}rocal${CMAKE_SHARED_LIBRARY_SUFFIX}")
    foreach(_root IN LISTS _candidates)
        if(NOT EXISTS "${_root}/include/rocal/rocal_api.h")
            continue()
        endif()
        foreach(_libdir lib lib64)
            if(EXISTS "${_root}/${_libdir}/${_library_name}")
                if(ROCAL_ROOT AND NOT _root STREQUAL ROCAL_ROOT)
                    message("-- ${Yellow}rocAL not found under ROCAL_ROOT=${ROCAL_ROOT}; using ${_root}${ColourReset}")
                endif()
                message("-- ${White}${PROJECT_NAME}: using rocAL from ${_root}${ColourReset}")
                set(ROCAL_FOUND TRUE PARENT_SCOPE)
                set(ROCAL_ROOT "${_root}" PARENT_SCOPE)
                set(ROCAL_LIBRARY "${_root}/${_libdir}/${_library_name}" PARENT_SCOPE)
                set(ROCAL_INCLUDE_DIR "${_root}/include/rocal" PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endforeach()
    message("-- ${Yellow}rocAL not found under ${_candidates}; linking rocal from the library search path${ColourReset}")
    set(ROCAL_FOUND FALSE PARENT_SCOPE)
    set(ROCAL_LIBRARY rocal PARENT_SCOPE)
    set(ROCAL_INCLUDE_DIR "${ROCM_PATH}/include/rocal" PARENT_SCOPE)
endfunction()
