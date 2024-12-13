################################################################################
# 
# MIT License
# 
# Copyright (c) 2024 Advanced Micro Devices, Inc.
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
find_path(ROCDECODE_INCLUDE_DIRS
    NAMES rocdecode.h rocparser.h
    HINTS
    $ENV{rocDecode_PATH}/include/rocdecode
    PATHS
    ${rocDecode_PATH}/include/rocdecode
    /usr/local/include/
    ${ROCM_PATH}/include/rocdecode
)
mark_as_advanced(ROCDECODE_INCLUDE_DIRS)

find_library(ROCDECODE_LIBRARIES
    NAMES rocdecode
    HINTS
    $ENV{rocDecode_PATH}/lib
    $ENV{rocDecode_PATH}/lib64
    PATHS
    ${rocDecode_PATH}/lib
    ${rocDecode_PATH}/lib64
    /usr/local/lib
    ${ROCM_PATH}/lib
)
mark_as_advanced(ROCDECODE_LIBRARIES)

if(ROCDECODE_LIBRARIES AND ROCDECODE_INCLUDE_DIRS)
    set(ROCDECODE_FOUND TRUE)
endif( )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rocDecode
    FOUND_VAR ROCDECODE_FOUND
    REQUIRED_VARS
        ROCDECODE_INCLUDE_DIRS
        ROCDECODE_LIBRARIES
)

set(ROCDECODE_FOUND ${ROCDECODE_FOUND} CACHE INTERNAL "")
set(ROCDECODE_LIBRARIES ${ROCDECODE_LIBRARIES} CACHE INTERNAL "")
set(ROCDECODE_INCLUDE_DIRS ${ROCDECODE_INCLUDE_DIRS} CACHE INTERNAL "")

if(ROCDECODE_FOUND)
    message("-- ${White}Using rocDecode -- \n\tLibraries:${ROCDECODE_LIBRARIES} \n\tIncludes:${ROCDECODE_INCLUDE_DIRS}${ColourReset}")
else()
    if(rocDecode_FIND_REQUIRED)
        message(FATAL_ERROR "{Red}FindrocDecode -- NOT FOUND${ColourReset}")
    endif()
    message( "-- ${Yellow}NOTE: FindrocDecode failed to find -- rocDecode${ColourReset}" )
endif()
