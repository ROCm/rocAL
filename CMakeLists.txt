# Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
cmake_minimum_required(VERSION 3.5)

project(rocAL)
set(VERSION "1.0.0")

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

find_program(MAKE_NSIS_EXE makensis)
find_program(RPMBUILD_EXE rpmbuild)
find_program(DPKG_EXE dpkg)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# avoid setting the default installation path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "rocAL default installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# rocAL Default Options
option(ENHANCED_MESSAGE "rocAL Enhanced Message Option"        ON)
option(GPU_SUPPORT      "Build rocAL with GPU Support"         ON)
option(BUILD_DEV        "Build rocAL Development Package"      ON)
option(AMD_FP16_SUPPORT "Build rocAL with float16 Support"    OFF)
option(BUILD_PYPACKAGE  "Build rocAL Python Package"           ON)
option(BUILD_WITH_AMD_ADVANCE "Build rocAL for advanced AMD GPU Architecture"    OFF)

set(DEFAULT_BUILD_TYPE "Release")

if(ENHANCED_MESSAGE)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(BoldBlue    "${Esc}[1;34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
endif()
message("-- ${BoldBlue}rocAL Version -- ${VERSION}${ColourReset}")
message("-- ${BoldBlue}rocAL Install Path -- ${CMAKE_INSTALL_PREFIX}${ColourReset}")

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
  set(GPU_SUPPORT OFF)
  set(BACKEND "CPU")
  message("-- ${Magenta}Apple macOS Detected -- GPU Support turned OFF${ColourReset}")
else()
  # Changes for RPATH Removal from Binaries:
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
  set(CMAKE_SKIP_INSTALL_RPATH TRUE)
endif()

if(NOT DEFINED BACKEND)
  set(BACKEND "HIP")   # set default backend to HIP
else()
  if("${BACKEND}" STREQUAL "OPENCL" OR "${BACKEND}" STREQUAL "OCL" OR "${BACKEND}" STREQUAL "OpenCL")
    set(BACKEND "OPENCL")
  elseif("${BACKEND}" STREQUAL "HIP" OR "${BACKEND}" STREQUAL "hip")
    set(BACKEND "HIP")
  elseif("${BACKEND}" STREQUAL "host" OR "${BACKEND}" STREQUAL "CPU" OR "${BACKEND}" STREQUAL "cpu")
    set(BACKEND "CPU")
    set(GPU_SUPPORT OFF)
  else()
    message("-- ${Red}Warning: rocAL backend option: ${BACKEND} -- unknown${ColourReset}")
    message("-- ${Red}Warning: rocAL default backend will be enforced${ColourReset}")
    set(BACKEND "HIP")
  endif()
endif()
message("-- ${BoldBlue}rocAL Backend set to -- ${BACKEND}${ColourReset}")

# rocAL Build Type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "rocAL Default Build Type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
if(CMAKE_BUILD_TYPE MATCHES Debug)
  # -O0  -- Don't Optimize output file
  # -g -- generate debugging information
  # -Og -- Optimize for debugging experience rather than speed or size
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Og")
else()
  # -O3 -- Optimize output file
  # -DNDEBUG -- turn off asserts
  # -fPIC -- Generate position-independent code if possible
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")
endif()
message("-- ${BoldBlue}rocAL Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")

message("-- ${Cyan}rocAL Developer Options${ColourReset}")
message("-- ${Cyan}     -D GPU_SUPPORT=${GPU_SUPPORT} [Turn ON/OFF GPU support (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D BACKEND=${BACKEND} [Select rocAL Backend [options:CPU/OPENCL/HIP](default:HIP)]${ColourReset}")
message("-- ${Cyan}     -D BUILD_DEV=${BUILD_DEV} [rocAL Developement/Runtime Build(default:ON)]${ColourReset}")
message("-- ${Cyan}     -D AMD_FP16_SUPPORT=${AMD_FP16_SUPPORT} [Turn ON/OFF OpenVX FP16 Support (default:OFF)]${ColourReset}")
message("-- ${Cyan}     -D BUILD_PYPACKAGE=${BUILD_PYPACKAGE} [rocAL Python Package(default:ON)]${ColourReset}")

if(AMD_FP16_SUPPORT)
  add_definitions(-DAMD_FP16_SUPPORT)
  message("-- ${Blue}rocAL -- -DAMD_FP16_SUPPORT definition added${ColourReset}")
endif(AMD_FP16_SUPPORT)

add_subdirectory(rocAL)
add_subdirectory(rocAL_pybind)

# install rocAL docs -- {ROCM_PATH}/${CMAKE_INSTALL_DATADIR}/doc/rocal/
install(FILES docs/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/rocal)

enable_testing()
include(CTest)
add_subdirectory(tests/cpp_api_tests)

# set package information
set(CPACK_PACKAGE_VERSION       ${VERSION})
set(CPACK_PACKAGE_NAME          "rocal")
set(CPACK_PACKAGE_LICENSE       "MIT")
set(CPACK_PACKAGE_CONTACT       "rocAL Support <mivisionx.support@amd.com>")
set(CPACK_PACKAGE_VENDOR        "AMD Radeon")
set(CPACK_PACKAGE_ICON          "${CMAKE_SOURCE_DIR}/docs/data/rocAL_logo.png")
set(CPACK_PACKAGE_GROUP         "Development/Tools")
set(CPACK_PACKAGE_HOMEPAGE      "https://github.com/ROCmSoftwarePlatform/rocAL")

set(CPACK_DEBIAN_PACKAGE_DESCRIPTION   "AMD rocAL is a comprehensive Augmentation Library
  The AMD ROCm Augmentation Library (rocAL) is designed to efficiently decode and process images \
and videos from a variety of storage formats and modify them through a processing graph programmable by the user.")
set(CPACK_RPM_PACKAGE_SUMMARY      "AMD rocAL is a comprehensive Augmentation Library")

# set license information
set(CPACK_RPM_PACKAGE_LICENSE    "MIT")
set(CPACK_RESOURCE_FILE_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/rocal)

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
endif()

set(CPACK_DEBIAN_FILE_NAME       "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME          "RPM-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
set(CPACK_RPM_PACKAGE_RELEASE    "local")

if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
endif()

# set dependency to ROCm if set to TRUE, default to OFF
set(ROCM_DEP_ROCMCORE OFF CACHE BOOL "Set rocm-core dependency")
if(ROCM_DEP_ROCMCORE)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "mivisionx rocm-core")
  set(CPACK_RPM_PACKAGE_REQUIRES   "mivisionx rocm-core")
endif()

# '%{?dist}' breaks manual builds on debian systems due to empty Provides
execute_process(COMMAND rpm --eval %{?dist}
                RESULT_VARIABLE PROC_RESULT
                OUTPUT_VARIABLE EVAL_RESULT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if (PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
  string (APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()

set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

set(CPACK_DEBIAN_PACKAGE_HOMEPAGE   ${CPACK_PACKAGE_HOMEPAGE})
set(CPACK_RPM_PACKAGE_URL           ${CPACK_PACKAGE_HOMEPAGE})
set(CPACK_RPM_PACKAGE_AUTOREQPROV   "no")

if(APPLE)
  set(CPACK_GENERATOR "Bundle")
  set(CPACK_BUNDLE_NAME ${CPACK_PACKAGE_NAME})
  set(CPACK_BUNDLE_ICON ${CPACK_PACKAGE_ICON})
  set(CPACK_BUNDLE_PLIST "${CMAKE_SOURCE_DIR}/.github/info.plist")
else()
  set(CPACK_GENERATOR "TGZ;ZIP")
  if(EXISTS ${MAKE_NSIS_EXE})
    list(APPEND CPACK_GENERATOR "NSIS")
  endif()
  if(EXISTS ${RPMBUILD_EXE})
    list(APPEND CPACK_GENERATOR "RPM")
  endif()
  if(EXISTS ${DPKG_EXE})
    list(APPEND CPACK_GENERATOR "DEB")
  endif()
endif()

include(CPack)
