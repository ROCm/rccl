# MIT License
#
# Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

# Dependencies

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# GIT

# Test dependencies

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

if(NOT INSTALL_DEPENDENCIES)
    find_package(GTest 1.11)
endif()

if(NOT GTest_FOUND AND BUILD_TESTS OR INSTALL_DEPENDENCIES)
    if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
        # hip-clang cannot compile googlebenchmark for some reason
        set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=g++")
    endif()

#       unset(GTEST_INCLUDE_DIR CACHE)
#	unset(GTEST_INCLUDE_DIRS CACHE)
    message(STATUS "GTest not found. Downloading and building GTest.")
    # Download, build and install googletest library
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")

    download_project(PROJ                googletest
                     GIT_REPOSITORY      https://github.com/google/googletest.git
                     GIT_TAG             release-1.11.0
                     INSTALL_DIR         ${GTEST_ROOT}
                     CMAKE_ARGS          -DBUILD_GTEST=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${COMPILER_OVERRIDE} -DBUILD_SHARED_LIBS=OFF
                     LOG_DOWNLOAD        TRUE
                     LOG_CONFIGURE       TRUE
                     LOG_BUILD           TRUE
                     LOG_INSTALL         TRUE
                     UPDATE_DISCONNECTED TRUE
    )
    set(GTEST_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/gtest/include CACHE PATH "")
    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/gtest/lib)
        set(GTEST_BOTH_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/gtest/lib/libgtest.a;${CMAKE_CURRENT_BINARY_DIR}/gtest/lib/libgtest_main.a CACHE PATH "")
    elseif(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/gtest/lib64)
        set(GTEST_BOTH_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/gtest/lib64/libgtest.a;${CMAKE_CURRENT_BINARY_DIR}/gtest/lib64/libgtest_main.a CACHE PATH "")
    else()
        message(FATAL_ERROR "Cannot find gtest library installation path.")
    find_package(GTest REQUIRED CONFIG PATHS ${GTEST_ROOT})
    endif()
endif()


# Find or download/install rocm-cmake project
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package(ROCM 0.7.3 QUIET CONFIG PATHS /opt/rocm)
if(NOT ROCM_FOUND)
    set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
    file(
        DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
        ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
        STATUS rocm_cmake_download_status LOG rocm_cmake_download_log
    )
    list(GET rocm_cmake_download_status 0 rocm_cmake_download_error_code)
    if(rocm_cmake_download_error_code)
        message(FATAL_ERROR "Error: downloading "
            "https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip failed "
            "error_code: ${rocm_cmake_download_error_code} "
            "log: ${rocm_cmake_download_log} "
        )
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
        WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}
        RESULT_VARIABLE rocm_cmake_unpack_error_code
    )
    execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
      WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
    execute_process( COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
      WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

    if(rocm_cmake_unpack_error_code)
        message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip failed")
    endif()
    find_package( ROCM 0.7.3 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMClients)
include( ROCMHeaderWrapper )
