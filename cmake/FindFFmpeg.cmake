################################################################################
# 
# MIT License
# 
# Copyright (c) 2017 - 2026 Advanced Micro Devices, Inc.
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
# - Try to find ffmpeg libraries (libavcodec, libavformat, libavutil and libswscale)
# Once done this will define
#
# FFMPEG_FOUND - system has ffmpeg or libav
# FFMPEG_INCLUDE_DIR - the ffmpeg include directory
# FFMPEG_LIBRARIES - Link these to use ffmpeg
# FFMPEG_VERSION - detected FFmpeg version string
################################################################################

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig")

# Use pkg-config to get the directories and version hints, then use these values
# in the FIND_PATH() and FIND_LIBRARY() calls.
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(_FFMPEG_AVCODEC libavcodec QUIET)
  pkg_check_modules(_FFMPEG_AVFORMAT libavformat QUIET)
  pkg_check_modules(_FFMPEG_AVUTIL libavutil QUIET)
  pkg_check_modules(_FFMPEG_SWSCALE libswscale QUIET)
endif()

# AVCODEC
find_path(AVCODEC_INCLUDE_DIR 
  NAMES libavcodec/avcodec.h
  PATHS ${_FFMPEG_AVCODEC_INCLUDE_DIRS}
    /usr/local/include
    /usr/include
    /opt/local/include
    /sw/include
  PATH_SUFFIXES ffmpeg libav
)
mark_as_advanced(AVCODEC_INCLUDE_DIR)
find_library(AVCODEC_LIBRARY
  NAMES avcodec
  PATHS ${_FFMPEG_AVCODEC_LIBRARY_DIRS}
    /usr/local/lib
    /usr/lib
    /opt/local/lib
    /sw/lib
)
mark_as_advanced(AVCODEC_LIBRARY)

# AVFORMAT
find_path(AVFORMAT_INCLUDE_DIR 
  NAMES libavformat/avformat.h
  PATHS ${_FFMPEG_AVFORMAT_INCLUDE_DIRS}
    /usr/local/include
    /usr/include
    /opt/local/include
    /sw/include
  PATH_SUFFIXES ffmpeg libav
)
mark_as_advanced(AVFORMAT_INCLUDE_DIR)
find_library(AVFORMAT_LIBRARY
  NAMES avformat
  PATHS ${_FFMPEG_AVFORMAT_LIBRARY_DIRS}
    /usr/local/lib
    /usr/lib
    /opt/local/lib
    /sw/lib
)
mark_as_advanced(AVFORMAT_LIBRARY)

# AVUTIL
find_path(AVUTIL_INCLUDE_DIR 
  NAMES libavutil/avutil.h
  PATHS ${_FFMPEG_AVUTIL_INCLUDE_DIRS}
    /usr/local/include
    /usr/include
    /opt/local/include
    /sw/include
  PATH_SUFFIXES ffmpeg libav
)
mark_as_advanced(AVUTIL_INCLUDE_DIR)
find_library(AVUTIL_LIBRARY
  NAMES avutil
  PATHS ${_FFMPEG_AVUTIL_LIBRARY_DIRS}
    /usr/local/lib
    /usr/lib
    /opt/local/lib
    /sw/lib
)
mark_as_advanced(AVUTIL_LIBRARY)

# SWSCALE  
find_path(SWSCALE_INCLUDE_DIR 
  NAMES libswscale/swscale.h
  PATHS ${_FFMPEG_SWSCALE_INCLUDE_DIRS}
    /usr/local/include
    /usr/include
    /opt/local/include
    /sw/include
  PATH_SUFFIXES ffmpeg libav sw
)
mark_as_advanced(SWSCALE_INCLUDE_DIR)
find_library(SWSCALE_LIBRARY
  NAMES swscale
  PATHS ${_FFMPEG_SWSCALE_LIBRARY_DIRS}
    /usr/local/lib
    /usr/lib
    /opt/local/lib
    /sw/lib
)
mark_as_advanced(SWSCALE_LIBRARY)

# Read <name>/version.h (and version_major.h, which newer FFmpeg uses for the MAJOR define)
# under <include_dir> into <out_var> as MAJOR.MINOR.MICRO, or leave it empty.
function(_ffmpeg_header_version name include_dir out_var)
  set(${out_var} "" PARENT_SCOPE)
  string(TOUPPER "${name}" _upper)
  set(_version_header "${include_dir}/${name}/version.h")
  if(NOT include_dir OR NOT EXISTS "${_version_header}")
    return()
  endif()
  set(_major_header "${_version_header}")
  if(EXISTS "${include_dir}/${name}/version_major.h")
    set(_major_header "${include_dir}/${name}/version_major.h")
  endif()
  set(_parts "")
  foreach(_part MAJOR MINOR MICRO)
    set(_header "${_version_header}")
    if(_part STREQUAL "MAJOR")
      set(_header "${_major_header}")
    endif()
    file(STRINGS "${_header}" _line REGEX "^#define[ \t]+${_upper}_VERSION_${_part}[ \t]+[0-9]+")
    string(REGEX REPLACE "^#define[ \t]+${_upper}_VERSION_${_part}[ \t]+([0-9]+).*" "\\1" _value "${_line}")
    if(NOT _value MATCHES "^[0-9]+$")
      return()
    endif()
    list(APPEND _parts "${_value}")
  endforeach()
  string(REPLACE ";" "." _version "${_parts}")
  set(${out_var} "${_version}" PARENT_SCOPE)
endfunction()

# Derive the per-component versions from the headers that will be compiled, re-read on every
# configure, so a changed FFmpeg install is picked up without clearing the cache.
# pkg-config is only a fallback, and it is queried directly: pkg_check_modules does not re-query
# a module it found before, so its cached version can describe an FFmpeg that is no longer
# installed. A component found by neither is left empty, never at an earlier configure's value.
# The results are plain variables. find_package(FFmpeg) runs in the top-level scope (and in the
# video plugin when it is built on its own), so they reach every consumer without a cache entry
# that could outlive the install.
_ffmpeg_header_version(libavcodec "${AVCODEC_INCLUDE_DIR}" _FFMPEG_AVCODEC_HEADER_VERSION)
_ffmpeg_header_version(libavformat "${AVFORMAT_INCLUDE_DIR}" _FFMPEG_AVFORMAT_HEADER_VERSION)
_ffmpeg_header_version(libavutil "${AVUTIL_INCLUDE_DIR}" _FFMPEG_AVUTIL_HEADER_VERSION)
foreach(_component AVCODEC AVFORMAT AVUTIL)
  if(_FFMPEG_${_component}_HEADER_VERSION)
    set(_FFMPEG_${_component}_VERSION "${_FFMPEG_${_component}_HEADER_VERSION}")
  elseif(PKG_CONFIG_FOUND AND _FFMPEG_${_component}_FOUND)
    string(TOLOWER "lib${_component}" _module)
    execute_process(COMMAND ${PKG_CONFIG_EXECUTABLE} --modversion ${_module}
      OUTPUT_VARIABLE _FFMPEG_${_component}_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _result ERROR_QUIET)
    if(NOT _result EQUAL 0)
      set(_FFMPEG_${_component}_VERSION "")
    endif()
  else()
    set(_FFMPEG_${_component}_VERSION "")
  endif()
endforeach()
set(FFMPEG_VERSION "${_FFMPEG_AVCODEC_VERSION}")

# Export the same per-component version variables the rest of the build expects.
# If neither the headers nor pkg-config provided one, approximate it with the libavcodec version.
if(NOT _FFMPEG_AVFORMAT_VERSION AND FFMPEG_VERSION)
  set(_FFMPEG_AVFORMAT_VERSION "${FFMPEG_VERSION}")
endif()
if(NOT _FFMPEG_AVUTIL_VERSION AND FFMPEG_VERSION)
  set(_FFMPEG_AVUTIL_VERSION "${FFMPEG_VERSION}")
endif()

if(AVCODEC_LIBRARY AND AVFORMAT_LIBRARY AND AVUTIL_LIBRARY AND SWSCALE_LIBRARY AND
   AVCODEC_INCLUDE_DIR AND AVFORMAT_INCLUDE_DIR AND AVUTIL_INCLUDE_DIR AND SWSCALE_INCLUDE_DIR)
  set(FFMPEG_FOUND TRUE)
  set(FFMPEG_INCLUDE_DIR "${AVFORMAT_INCLUDE_DIR}" CACHE INTERNAL "FFmpeg include directory")
  set(FFMPEG_LIBRARIES 
    "${AVCODEC_LIBRARY}"
    "${AVFORMAT_LIBRARY}"
    "${AVUTIL_LIBRARY}"
    "${SWSCALE_LIBRARY}"
    CACHE INTERNAL "FFmpeg libraries"
  )
else()
  set(FFMPEG_FOUND FALSE)
endif()

# Reject FFmpeg older than 4.0.4 before find_package_handle_standard_args. That helper
# sets FFMPEG_FOUND from the library list alone, so a rejection made before it is overwritten
# and an old host is accepted. The searches above must stay first: the video plugin calls
# find_package(FFmpeg REQUIRED), and the helper aborts when its required variables are still empty.
set(FFMPEG_VERSION_REJECTED FALSE)
if(FFMPEG_FOUND)
  if(_FFMPEG_AVCODEC_VERSION VERSION_LESS 58.18.100 OR
     _FFMPEG_AVFORMAT_VERSION VERSION_LESS 58.12.100 OR
     _FFMPEG_AVUTIL_VERSION VERSION_LESS 56.14.100)
    message("-- ${White}FFMPEG   required min version - 4.0.4 Found:${FFMPEG_VERSION}${ColourReset}")
    message("-- ${White}AVCODEC  required min version - 58.18.100 Found:${_FFMPEG_AVCODEC_VERSION}${ColourReset}")
    message("-- ${White}AVFORMAT required min version - 58.12.100 Found:${_FFMPEG_AVFORMAT_VERSION}${ColourReset}")
    message("-- ${White}AVUTIL   required min version - 56.14.100 Found:${_FFMPEG_AVUTIL_VERSION}${ColourReset}")
    set(FFMPEG_FOUND FALSE)
    set(FFMPEG_VERSION_REJECTED TRUE)
    message("-- ${Yellow}NOTE: FindFFmpeg failed to find -- FFMPEG${ColourReset}")
  endif()
endif()

if(_FFMPEG_AVCODEC_VERSION VERSION_LESS 59.18.100 OR
   _FFMPEG_AVFORMAT_VERSION VERSION_LESS 59.16.100 OR
   _FFMPEG_AVUTIL_VERSION VERSION_LESS 57.17.100)
  set(FFMPEG_VERSION_4 TRUE CACHE INTERNAL "")
else()
  set(FFMPEG_VERSION_4 FALSE CACHE INTERNAL "")
endif()

# find_package() sets FFmpeg_FIND_REQUIRED and FFmpeg_FIND_QUIETLY from the package name.
if(FFMPEG_VERSION_REJECTED)
  if(FFmpeg_FIND_REQUIRED)
    message(FATAL_ERROR "${Red}FindFFmpeg -- FFmpeg ${FFMPEG_VERSION} is older than the minimum 4.0.4 (libavcodec >= 58.18.100, libavformat >= 58.12.100, libavutil >= 56.14.100). Found libavcodec ${_FFMPEG_AVCODEC_VERSION}, libavformat ${_FFMPEG_AVFORMAT_VERSION}, libavutil ${_FFMPEG_AVUTIL_VERSION}.${ColourReset}")
  endif()
else()
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    FFmpeg
    FOUND_VAR FFMPEG_FOUND
    REQUIRED_VARS
      FFMPEG_LIBRARIES
      FFMPEG_INCLUDE_DIR
      AVCODEC_INCLUDE_DIR
      AVCODEC_LIBRARY
      AVFORMAT_INCLUDE_DIR
      AVFORMAT_LIBRARY
      AVUTIL_INCLUDE_DIR
      AVUTIL_LIBRARY
      SWSCALE_INCLUDE_DIR
      SWSCALE_LIBRARY
    VERSION_VAR FFMPEG_VERSION
  )
endif()

if(FFMPEG_FOUND)
  if(NOT FFmpeg_FIND_QUIETLY)
    message("-- ${White}Using FFMPEG -- \n\tLibraries:${FFMPEG_LIBRARIES} \n\tIncludes:${FFMPEG_INCLUDE_DIR}${ColourReset}")
  endif()
else()
  if(FFmpeg_FIND_REQUIRED AND NOT FFMPEG_VERSION_REJECTED)
    message(FATAL_ERROR "${Red}FindFFmpeg -- libavcodec or libavformat or libavutil or libswscale NOT FOUND${ColourReset}")
  endif()
endif()
