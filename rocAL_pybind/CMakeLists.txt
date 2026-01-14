# Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.13)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- rocAL: INFO - ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
elseif(NOT DEFINED CMAKE_CXX_COMPILER AND NOT EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
endif()

# Set Project Version and Language
project(rocal_pybind VERSION ${VERSION} LANGUAGES CXX)

set(TARGET_NAME rocal_pybind)

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# CMake Config location
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(BUILD_ROCAL_PYBIND true)
set(CMAKE_SKIP_BUILD_RPATH TRUE)

find_package(MIVisionX QUIET)
find_package(TurboJpeg QUIET)
find_package(pybind11 QUIET CONFIG)
find_package(HALF QUIET)
find_package(OpenMP COMPONENTS CXX QUIET)

# Backend
if(BUILD_PYPACKAGE)
    if(NOT DEFINED BACKEND)
        set(BACKEND "HIP")   # set default backend to HIP
    else()
        if("${BACKEND}" STREQUAL "HIP" OR "${BACKEND}" STREQUAL "hip")
            set(BACKEND "HIP")
        elseif("${BACKEND}" STREQUAL "host" OR "${BACKEND}" STREQUAL "CPU" OR "${BACKEND}" STREQUAL "cpu")
            set(BACKEND "CPU")
            set(GPU_SUPPORT OFF)
        else()
            message("-- ${Red}Warning: rocAL PyBind backend option: ${BACKEND} -- unknown${ColourReset}")
            message("-- ${Red}Warning: rocAL PyBind default backend will be enforced${ColourReset}")
            set(BACKEND "HIP")
        endif()
    endif()
    message("-- ${BoldBlue}rocAL PyBind Backend set to -- ${BACKEND}${ColourReset}")

    # Find HIP Support
    if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
        if(NOT DEFINED HIP_PATH)
            if(NOT DEFINED ENV{HIP_PATH})
                set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
            else()
                set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
            endif()
        endif()
        list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})
        find_package(HIP QUIET)
        if(HIP_FOUND)
            message("-- ${White}rocAL PyBind -- Using HIP -- Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER} ${ColourReset})
        else()
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Yellow}rocAL PyBind requires HIP, Not Found${ColourReset}")
        endif()
    endif()
endif()

# Dependency Check
# MIVisionX
if(NOT MIVisionX_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL PyBind library requires AMD's MIVisionX library, Not Found ${ColourReset}")
else()
    #find the MIVisionX backend type
    set(MIVISIONX_BACKEND_HIP_FOUND 0)
    if(EXISTS ${MIVisionX_INCLUDE_DIRS}/openvx_backend.h)
        file(READ ${MIVisionX_INCLUDE_DIRS}/openvx_backend.h MIVISIONX_BACKEND_FILE)
        string(REGEX MATCH "ENABLE_HIP ([0-9]*)" _ ${MIVISIONX_BACKEND_FILE})
        set(MIVISIONX_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    else()
        message(FATAL_ERROR "${Red}ERROR: ${MIVisionX_INCLUDE_DIRS}/openvx_backend.h file Not Found. MIVisionX Package REQUIRED ${ColourReset}")
    endif()

    if("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        if (NOT MIVISIONX_BACKEND_HIP_FOUND)
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Red}ERROR: rocAL PyBind -- MIVisionX with the HIP backend Not Found${ColourReset}")
        endif()
    endif()
endif()
# Turbo JPEG
if(NOT TurboJpeg_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL pybind requires TurboJpeg, Not Found${ColourReset}")
endif()

# PyBind11
if(NOT pybind11_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL pybind requires pybind11, Not Found${ColourReset}")
endif()
# Half
if(NOT HALF_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL pybind requires Half, Not Found${ColourReset}")
endif()
# OpenMP
if(NOT OpenMP_CXX_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL pybind library requires OpenMP, Not Found ${ColourReset}")
endif()

if(${BUILD_ROCAL_PYBIND})
    # multi version python support
    set(PYTHON_VERSIONS "3.13" "3.12" "3.11" "3.10" "3.9" "3.8")
    if(PYTHON_VERSION_SUGGESTED)
        set(PYTHON_VERSIONS "${PYTHON_VERSION_SUGGESTED}")
    endif()

    foreach(PY_VERSION ${PYTHON_VERSIONS})
        message(STATUS "Searching for Python version: ${PY_VERSION}")

        foreach(v IN ITEMS
            Python3_EXECUTABLE
            Python3_VERSION
            Python3_SOABI
            Python3_EXTENSION_SUFFIX
            Python3_Interpreter_FOUND
            Python3_Development_FOUND
            Python3_INCLUDE_DIRS
            Python3_LIBRARY
            Python3_LIBRARIES
            Python3_STDARCH
            Python3_FIND_IMPLEMENTATIONS
        )
            unset(${v} CACHE)
            unset(${v})
        endforeach()

        set(Python3_FIND_VIRTUALENV FIRST)

        find_package(Python3 "${PY_VERSION}" EXACT QUIET COMPONENTS Interpreter Development)
        set(PYTHON_FOUND_${PY_VERSION} ${Python3_FOUND})

        if(NOT PYTHON_FOUND_${PY_VERSION})
            message("-- ${Yellow}NOTE: Python${PY_VERSION} Not Found, not building rocAL pybind for Python${PY_VERSION}${ColourReset}")
            continue()
        else()
            # Enable PythonFunction operator in core rocAL library when building pybind
            # and all Python/pybind11/MIVisionX conditions are satisfied.
            if (VX_EXT_RPP_VERSION VERSION_GREATER_EQUAL "3.1.1")
                target_compile_definitions(rocal PUBLIC ROCAL_PYTHON_FUNCTION)
                target_include_directories(rocal PRIVATE ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})
                target_link_libraries(rocal ${Python3_LIBRARIES})
                message("-- ${White}rocAL PythonFunction operator enabled for pybind${ColourReset}")
            else()
                message("-- ${Yellow}NOTE: rocAL PythonFunction operator disabled - requires Python3, pybind11 and MIVisionX vx_rpp >= 3.1.1${ColourReset}")
            endif()


            message("-- ${Green}NOTE: Python ${PY_VERSION} Found, building rocAL pybind for ${Python3_EXECUTABLE}${ColourReset}")
            # ----- Create a unique pybind module target for this Python -----
            set(mod_tgt "rocal_pybind_${PY_VERSION}")                # CMake target name

            # avoid setting the default installation path to /usr/local
            if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
                set(CMAKE_INSTALL_PREFIX_PYTHON ${Python3_STDARCH} CACHE PATH "rocAL PyBind default installation path" FORCE)
            endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
            set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

            # Reset and build link library list for this specific Python version
            set(LINK_LIBRARY_LIST_${PY_VERSION} "")
            # MIVisionX and RPP
            list(APPEND LINK_LIBRARY_LIST_${PY_VERSION} ${OPENVX_LIBRARIES} ${VXRPP_LIBRARIES})
            # TurboJPEG
            list(APPEND LINK_LIBRARY_LIST_${PY_VERSION} ${TurboJpeg_LIBRARIES})
            # rocAL
            list(APPEND LINK_LIBRARY_LIST_${PY_VERSION} rocal)
            # OpenMP
            list(APPEND LINK_LIBRARY_LIST_${PY_VERSION} ${OpenMP_CXX_LIBRARIES})
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath='$ORIGIN:$ORIGIN/llvm/lib'")

            # Binding sources
            set(BINDING_SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/rocal_pybind.cpp
                # add other .cpp files here explicitly
            )

            pybind11_add_module(${mod_tgt} MODULE ${BINDING_SOURCES})
            # Use target-specific includes to avoid cross-contamination between Python versions
            target_include_directories(${mod_tgt} PRIVATE 
                ${MIVisionX_INCLUDE_DIRS}
                ${TurboJpeg_INCLUDE_DIRS}
                ${Python3_INCLUDE_DIRS}
                ${pybind11_INCLUDE_DIRS}
                ${HALF_INCLUDE_DIRS}
                ../rocAL/include/api/
            )
            target_link_libraries(${mod_tgt} PRIVATE ${LINK_LIBRARY_LIST_${PY_VERSION}} ${Python3_LIBRARIES})

            add_dependencies(${mod_tgt} rocal)
            message("-- ${White}rocal_pybind -- Link Libraries: ${LINK_LIBRARY_LIST_${PY_VERSION}}${ColourReset}")
            message("-- ${White}rocal_pybind -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")

            #DLPack
            find_package(DLPACK QUIET)
            if(DLPACK_FOUND)
                include_directories(${DLPACK_INCLUDE_DIRS})
                target_compile_definitions(${mod_tgt} PUBLIC ENABLE_DLPACK=1)
            else()
                target_compile_definitions(${mod_tgt} PUBLIC ENABLE_DLPACK=0)
                message("-- ${Yellow}NOTE: DLPack Not Found - rocAL Tensorflow and Generic plugins built without GPU support!${ColourReset}")
            endif()

            execute_process(
                COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
                OUTPUT_VARIABLE Python3_EXTENSION_SUFFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            set_target_properties(${mod_tgt} PROPERTIES
                PREFIX ""
                SUFFIX "${Python3_EXTENSION_SUFFIX}"
                OUTPUT_NAME "rocal_pybind")

            # Python files - Install
            install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/amd/"
                    DESTINATION ${CMAKE_INSTALL_LIBDIR}/amd
                    COMPONENT dev)

            # rocAL pybind libs - install
            install(TARGETS ${mod_tgt}
                    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                    COMPONENT dev)

            # PyBind Test - Install
            install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/pybind DESTINATION ${CMAKE_INSTALL_DATADIR}/rocal/test COMPONENT test)
            install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/python_api DESTINATION ${CMAKE_INSTALL_DATADIR}/rocal/test COMPONENT test)

            message("-- ${Green}Radeon Augmentation Library Python Binding - ${mod_tgt} module added ${ColourReset}")

        endif()
    endforeach()
endif()
