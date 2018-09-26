################################################################################
#
# MIT License
#
# Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
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

cmake_minimum_required( VERSION 2.8.12 )

set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")

project ( rccl C CXX)

find_package(ROCM REQUIRED PATHS /opt/rocm)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)

rocm_setup_version(
    VERSION 0.6.0-rc1
)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip /opt/rocm/hcc)

find_package(hip REQUIRED)

link_libraries(-amdgpu-target=gfx803 -amdgpu-target=gfx900 -amdgpu-target=gfx906)

option( BUILD_SHARED_LIBS "Build as a shared library" ON )

add_library( rccl
    src/rccl.cpp
    src/rcclAllReduce.cpp
    src/rcclBcast.cpp
    src/rcclReduce.cpp
    src/rcclTracker.cpp
    )

if( TARGET hip::device )
target_link_libraries( rccl PRIVATE hip::device )
target_link_libraries( rccl INTERFACE hip::host )
else()
target_link_libraries( rccl PUBLIC hip::hip_hcc ${hcc_LIBRARIES} )
endif()

rocm_install_targets(
  TARGETS rccl
  PREFIX rccl
  INCLUDE
    ${PROJECT_SOURCE_DIR}/inc
)

rocm_export_targets(
  NAMESPACE roc::
  PREFIX rccl
  TARGETS rccl
  DEPENDS hip
)

set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc")
set(CPACK_RPM_PACKAGE_REQUIRES "hip_hcc")

rocm_create_package(
    NAME rccl
    DESCRIPTION "AMD's multi-gpu MPI impelemtation"
    MAINTAINER "Aditya Aturi <Aditya.Atluri@amd.com>"
    LDCONFIG
)

rocm_install_symlink_subdir(rccl)

option(BUILD_DOC "Build Documentation" ON)

find_package(Doxygen)

if(DOXYGEN_FOUND)
    set(BUILD_DOC_DIR ${PROJECT_BINARY_DIR}/docs)
    if(NOT EXISTS ${BUILD_DOC_DIR})
        file(MAKE_DIRECTORY ${BUILD_DOC_DIR})
    endif()

    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    message("Doxygen build started")
    add_custom_target(Doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
else(DOXYGEN_FOUND)
    message("Doxygen needs to be installed to generate the documentation")
endif(DOXYGEN_FOUND)
