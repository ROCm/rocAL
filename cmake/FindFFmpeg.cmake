################################################################################
# 
# MIT License
# 
# Copyright (c) 2017 - 2025 Advanced Micro Devices, Inc.
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

# Derive a version string. Prefer the pkg-config value; if unavailable, fall
# back to parsing the libavcodec version header.
if(_FFMPEG_AVCODEC_VERSION)
  set(FFMPEG_VERSION "${_FFMPEG_AVCODEC_VERSION}")
elseif(AVCODEC_INCLUDE_DIR)
  if(EXISTS "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h")
    # Newer FFmpeg moved LIBAVCODEC_VERSION_MAJOR into version_major.h, leaving MINOR/MICRO
    # in version.h. Read MAJOR from version_major.h when present so version detection does not
    # come up empty and fall back to the legacy compatibility branch on recent headers.
    set(_AVCODEC_MAJOR_HEADER "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h")
    if(EXISTS "${AVCODEC_INCLUDE_DIR}/libavcodec/version_major.h")
      set(_AVCODEC_MAJOR_HEADER "${AVCODEC_INCLUDE_DIR}/libavcodec/version_major.h")
    endif()
    file(STRINGS "${_AVCODEC_MAJOR_HEADER}" AVCODEC_VERSION_MAJOR_REGEXP
         REGEX "^#define[ \t]+LIBAVCODEC_VERSION_MAJOR[ \t]+")
    file(STRINGS "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h" AVCODEC_VERSION_MINOR_REGEXP
         REGEX "^#define[ \t]+LIBAVCODEC_VERSION_MINOR[ \t]+")
    file(STRINGS "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h" AVCODEC_VERSION_MICRO_REGEXP
         REGEX "^#define[ \t]+LIBAVCODEC_VERSION_MICRO[ \t]+")
    string(REGEX REPLACE "^#define[ \t]+LIBAVCODEC_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1"
           AVCODEC_VERSION_MAJOR "${AVCODEC_VERSION_MAJOR_REGEXP}")
    string(REGEX REPLACE "^#define[ \t]+LIBAVCODEC_VERSION_MINOR[ \t]+([0-9]+).*" "\\1"
           AVCODEC_VERSION_MINOR "${AVCODEC_VERSION_MINOR_REGEXP}")
    string(REGEX REPLACE "^#define[ \t]+LIBAVCODEC_VERSION_MICRO[ \t]+([0-9]+).*" "\\1"
           AVCODEC_VERSION_MICRO "${AVCODEC_VERSION_MICRO_REGEXP}")
    if(AVCODEC_VERSION_MAJOR AND AVCODEC_VERSION_MINOR AND AVCODEC_VERSION_MICRO)
      set(FFMPEG_VERSION "${AVCODEC_VERSION_MAJOR}.${AVCODEC_VERSION_MINOR}.${AVCODEC_VERSION_MICRO}")
    endif()
  endif()
endif()

# Export the same per-component version variables the rest of the build expects.
# If pkg-config did not provide them, approximate them with the derived version.
if(NOT _FFMPEG_AVCODEC_VERSION AND FFMPEG_VERSION)
  set(_FFMPEG_AVCODEC_VERSION "${FFMPEG_VERSION}")
endif()
if(NOT _FFMPEG_AVFORMAT_VERSION AND FFMPEG_VERSION)
  set(_FFMPEG_AVFORMAT_VERSION "${FFMPEG_VERSION}")
endif()
if(NOT _FFMPEG_AVUTIL_VERSION AND FFMPEG_VERSION)
  set(_FFMPEG_AVUTIL_VERSION "${FFMPEG_VERSION}")
endif()
set(_FFMPEG_AVCODEC_VERSION "${_FFMPEG_AVCODEC_VERSION}" CACHE INTERNAL "Detected libavcodec version")

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
