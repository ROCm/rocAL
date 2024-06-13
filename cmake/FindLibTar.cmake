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
find_path(libtar_INCLUDE_DIRS
    NAMES libtar.h
    HINTS
    $ENV{LIBTAR_PATH}/include
    PATHS
    /usr/include
    /usr/local/include
)
mark_as_advanced(libtar_INCLUDE_DIRS)

find_library(libtar_LIBRARIES
        NAMES libtar.a tar libtar
        HINTS
        $ENV{LIBTAR_PATH}/lib
        $ENV{LIBTAR_PATH}/lib64
        PATHS ${CMAKE_SYSTEM_PREFIX_PATH} ${LIBTAR_PATH} "/usr/local" "/usr/lib"
        PATH_SUFFIXES lib lib64)

mark_as_advanced(libtar_LIBRARIES)

if(libtar_LIBRARIES AND libtar_INCLUDE_DIRS)
    message("-- ${Yellow}NOTE: rocAL built WITH LibTar - WebDataset Functionalities will be supported${ColourReset}")
    set(LIBTAR_FOUND TRUE)
else()
    message("---not found!!!!!!!!!!!!!!!!!!!!!!${ColourReset}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibTar
    FOUND_VAR  LIBTAR_FOUND 
    REQUIRED_VARS
        libtar_LIBRARIES
        libtar_INCLUDE_DIRS
)

set(LIBTAR_FOUND ${LIBTAR_FOUND} CACHE INTERNAL "")
set(libtar_LIBRARIES ${libtar_LIBRARIES} CACHE INTERNAL "")
set(libtar_INCLUDE_DIRS ${libtar_INCLUDE_DIRS} CACHE INTERNAL "")

if(LIBTAR_FOUND)
    message("-- ${Blue}Using Libtar -- \n\tLibraries:${libtar_LIBRARIES} \n\tIncludes:${libtar_INCLUDE_DIRS}${ColourReset}")   
else()
    message( "-- ${Yellow}NOTE: FindLibTar failed to find -- LibTar${ColourReset}" )
endif()
