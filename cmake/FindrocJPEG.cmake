################################################################################
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

# find rocJPEG - library and headers
find_path(ROCJPEG_INCLUDE_DIR NAMES rocjpeg.h PATHS ${ROCM_PATH}/include/rocjpeg)
find_library(ROCJPEG_LIBRARY NAMES rocjpeg HINTS ${ROCM_PATH}/lib)
mark_as_advanced(ROCJPEG_INCLUDE_DIR)
mark_as_advanced(ROCJPEG_LIBRARY)

if(ROCJPEG_INCLUDE_DIR AND ROCJPEG_LIBRARY)
    message("-- ${White}FindrocJPEG -- Using rocJPEG: \n\tIncludes:${ROCJPEG_INCLUDE_DIR}\n\tLib:${ROCJPEG_LIBRARY}${ColourReset}")
    set(ROCJPEG_FOUND TRUE)
else()
    if(rocJPEG_FIND_REQUIRED)
        message(FATAL_ERROR "FindrocJPEG -- Failed to find rocJPEG Library")
    endif()
    message( "-- ${Yellow}NOTE: FindrocJPEG failed to find rocJPEG -- INSTALL rocJPEG${ColourReset}" )
endif()

if(ROCJPEG_FOUND)
    # Find rocJPEG Version
    file(READ "${ROCJPEG_INCLUDE_DIR}/rocjpeg_version.h" ROCJPEG_VERSION_FILE)
    string(REGEX MATCH "ROCJPEG_MAJOR_VERSION ([0-9]*)" _ ${ROCJPEG_VERSION_FILE})
    set(ROCJPEG_VER_MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "ROCJPEG_MINOR_VERSION ([0-9]*)" _ ${ROCJPEG_VERSION_FILE})
    set(ROCJPEG_VER_MINOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "ROCJPEG_MICRO_VERSION ([0-9]*)" _ ${ROCJPEG_VERSION_FILE})
    set(ROCJPEG_VER_MICRO ${CMAKE_MATCH_1})
    message("-- ${White}Found rocJPEG Version: ${ROCJPEG_VER_MAJOR}.${ROCJPEG_VER_MINOR}.${ROCJPEG_VER_MICRO}${ColourReset}")
    mark_as_advanced(ROCJPEG_VER_MAJOR)
    mark_as_advanced(ROCJPEG_VER_MINOR)
    mark_as_advanced(ROCJPEG_VER_MICRO)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    rocJPEG
    FOUND_VAR
    ROCJPEG_FOUND
    REQUIRED_VARS
    ROCJPEG_INCLUDE_DIR
    ROCJPEG_LIBRARY
)

set(ROCJPEG_FOUND ${ROCJPEG_FOUND} CACHE INTERNAL "")
set(ROCJPEG_INCLUDE_DIR ${ROCJPEG_INCLUDE_DIR} CACHE INTERNAL "")
set(ROCJPEG_LIBRARY ${ROCJPEG_LIBRARY} CACHE INTERNAL "")
set(ROCJPEG_VER_MAJOR ${ROCJPEG_VER_MAJOR} CACHE INTERNAL "")
set(ROCJPEG_VER_MINOR ${ROCJPEG_VER_MINOR} CACHE INTERNAL "")
set(ROCJPEG_VER_MICRO ${ROCJPEG_VER_MICRO} CACHE INTERNAL "")