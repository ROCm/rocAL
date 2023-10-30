################################################################################
# 
# MIT License
# 
# Copyright (c) 2017 - 2023 Advanced Micro Devices, Inc.
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

if(TARGET std::filesystem)
    # This module has already been processed. Don't do it again.
    return()
endif()

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)

cmake_push_check_state()

set(CMAKE_REQUIRED_QUIET ${StdFilesystem_FIND_QUIETLY})

set(_found FALSE)

foreach(_candidate IN ITEMS filesystem;experimental/filesystem)
    set(_fs_header ${_candidate})
    string(REPLACE "/" "::" _fs_namespace ${_fs_header})
    set(_fs_namespace std::${_fs_namespace})

    string(CONFIGURE [[
        #include <@_fs_header@>

        int main() {
        auto cwd = @_fs_namespace@::current_path();
        return static_cast<int>(cwd.string().size());
        }
    ]] code @ONLY)

    set(_COMBO "")
    if("${_fs_header}" MATCHES .*experimental.*)
        string(APPEND _COMBO _EXPT)
    endif()

    # Try to compile a simple filesystem program without any linker flags
    check_cxx_source_compiles("${code}" CXX_FILESYSTEM_NO_LINK${_COMBO}_NEEDED)

    set(can_link ${CXX_FILESYSTEM_NO_LINK${_COMBO}_NEEDED})
    set(prev_libraries ${CMAKE_REQUIRED_LIBRARIES})

    if(NOT can_link)
        # Try linking the libstdc++ library
        set(CMAKE_REQUIRED_LIBRARIES ${prev_libraries} -lstdc++fs)
        check_cxx_source_compiles("${code}" CXX_FILESYSTEM_STDCPPFS${_COMBO}_NEEDED)
        set(can_link ${CXX_FILESYSTEM_STDCPPFS${_COMBO}_NEEDED})
    endif()

    if(NOT can_link)
        # Try linking the libc++ library
        set(CMAKE_REQUIRED_LIBRARIES ${prev_libraries} -lc++fs)
        check_cxx_source_compiles("${code}" CXX_FILESYSTEM_CPPFS${_COMBO}_NEEDED)
        set(can_link ${CXX_FILESYSTEM_CPPFS${_COMBO}_NEEDED})
    endif()

    if(NOT can_link)
        # Try linking the libc++experimental library
        set(CMAKE_REQUIRED_LIBRARIES ${prev_libraries} -lc++experimental)
        check_cxx_source_compiles("${code}" CXX_FILESYSTEM_CPPEXPERIMENTAL${_COMBO}_NEEDED)
        set(can_link ${CXX_FILESYSTEM_CPPEXPERIMENTAL${_COMBO}_NEEDED})
    endif()

    # Reset CMAKE_REQUIRED_LIBRARIES
    set(CMAKE_REQUIRED_LIBRARIES ${prev_libraries})
    unset(prev_libraries)

    if(can_link)
        add_library(std::filesystem INTERFACE IMPORTED)
        set(_found TRUE)

        if(CXX_FILESYSTEM_NO_LINK${_COMBO}_NEEDED)
            # Nothing to add...
        elseif(CXX_FILESYSTEM_STDCPPFS${_COMBO}_NEEDED)
            target_link_libraries(std::filesystem INTERFACE -lstdc++fs)
        elseif(CXX_FILESYSTEM_CPPFS${_COMBO}_NEEDED)
            target_link_libraries(std::filesystem INTERFACE -lc++fs)
        elseif(CXX_FILESYSTEM_CPPEXPERIMENTAL${_COMBO}_NEEDED)
            target_link_libraries(std::filesystem INTERFACE -lc++experimental)
        endif()

        string(TOLOWER ${CMAKE_BUILD_TYPE} _find_stdfs_cmake_build_type)
        if("${_find_stdfs_cmake_build_type}" MATCHES "(debug|buildfast)")
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
                # Without at least -O1, the Intel compiler errors out with:
                # c++/${cxxversion}/experimental/fs_path.h:822:
                # undefined reference to
                #     `std::codecvt_utf8<char, 1114111ul,
                #     (std::codecvt_mode)0>::codecvt_utf8(unsigned long)'
                target_compile_options(std::filesystem BEFORE
                    INTERFACE -mGLOB_opt_level=1)
                message(STATUS
                    "Injecting -mGLOB_opt_level=1 for std::filesystem to get"
                    " around ${CMAKE_CXX_COMPILER_ID} compiler bug")
            endif()
        endif()
        unset(_find_stdfs_cmake_build_type)

        break()
    endif()

endforeach()

if(_found)
    target_compile_definitions(std::filesystem INTERFACE
        TCM_FS_HEADER=<${_fs_header}>
        TCM_FS_NAMESPACE=${_fs_namespace})
endif()

cmake_pop_check_state()

set(StdFilesystem_FOUND ${_found} CACHE BOOL "TRUE if we can compile and link a program using std::filesystem" FORCE)

if(StdFilesystem_FIND_REQUIRED AND NOT StdFilesystem_FOUND)
    message(FATAL_ERROR "Cannot Compile simple program using std::(experimental::)filesystem")
endif()