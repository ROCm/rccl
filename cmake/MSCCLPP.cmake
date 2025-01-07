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

function(mscclpp_cmake_arg NAME)
    string (REPLACE ";" "$<SEMICOLON>" ARG_VALUE "${${NAME}}") # Replace ; with non-escapable SEMICOLON symbol to avoid CMake errors
    string(STRIP "${ARG_VALUE}" ARG_VALUE) # Eliminate whitespace, reducing to empty string if necessary

    # Only add a cmake argument if it has a value
    set(${NAME}_ARG "-D${NAME}=\"${ARG_VALUE}\"" PARENT_SCOPE)
    if("${ARG_VALUE}" STREQUAL "")
        set(${NAME}_ARG "" PARENT_SCOPE)
    endif()
endfunction()


if(ENABLE_MSCCLPP)
    # Try to find the mscclpp install
    set(MSCCLPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/ext/mscclpp CACHE PATH "")
    execute_process(
        COMMAND mkdir -p ${MSCCLPP_ROOT}
    )
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(mscclpp_nccl)

    if(NOT mscclpp_nccl_FOUND)
        # Ensure the source code is checked out
        set(MSCCLPP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/mscclpp CACHE PATH "")
        if(NOT EXISTS ${MSCCLPP_SOURCE}/CMakeLists.txt)
            message(STATUS "Checking out microsoft/mscclpp")
            execute_process(
                COMMAND git submodule update --init --recursive
                WORKING_DIRECTORY ${MSCCLPP_SOURCE}
            )
        endif()
        execute_process(
           COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/cpx.patch
           WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )
	execute_process(
           COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/read-allred.patch
           WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )
        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/mscclpp_ibv_access_relaxed_ordering.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        message(STATUS "Building mscclpp only for gfx942.")

        mscclpp_cmake_arg(CMAKE_PREFIX_PATH)
        mscclpp_cmake_arg(CMAKE_INSTALL_RPATH_USE_LINK_PATH)
        mscclpp_cmake_arg(HIP_COMPILER)

        set(GFX942_VARIANT "gfx942")
        if(BUILD_ADDRESS_SANITIZER)
            set(GFX942_VARIANT "gfx942:xnack+")
        endif()

        download_project(PROJ                mscclpp_nccl
                         # GIT_REPOSITORY      https://github.com/microsoft/mscclpp.git
                         # GIT_TAG             1e82dd444fc1ed8b7add354eebaab8a94e67d5fc
                         INSTALL_DIR         ${MSCCLPP_ROOT}
                         CMAKE_ARGS          -DAMDGPU_TARGETS=${GFX942_VARIANT} -DGPU_TARGETS=${GFX942_VARIANT} -DBYPASS_GPU_CHECK=ON -DUSE_ROCM=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_APPS_NCCL=ON -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> "${CMAKE_PREFIX_PATH_ARG}" -DCMAKE_VERBOSE_MAKEFILE=1 "${CMAKE_INSTALL_RPATH_USE_LINK_PATH_ARG}" "${HIP_COMPILER_ARG}" -DFETCHCONTENT_SOURCE_DIR_JSON=${CMAKE_CURRENT_SOURCE_DIR}/ext-src/json
                         LOG_DOWNLOAD        FALSE
                         LOG_CONFIGURE       FALSE
                         LOG_BUILD           FALSE
                         LOG_INSTALL         FALSE
                         UPDATE_DISCONNECTED TRUE
                         SOURCE_DIR          ${MSCCLPP_SOURCE}
        )

        find_package(mscclpp_nccl REQUIRED)
	execute_process(
    		COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/cpx.patch
        	WORKING_DIRECTORY ${MSCCLPP_SOURCE}
    	)
	execute_process(
           COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/read-allred.patch
           WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )
        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/mscclpp_ibv_access_relaxed_ordering.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )
    endif()

    execute_process(COMMAND objcopy
                    --redefine-syms=${CMAKE_CURRENT_SOURCE_DIR}/src/misc/mscclpp/mscclpp_nccl_syms.txt
                    "${MSCCLPP_ROOT}/lib/libmscclpp_nccl_static.a"
                    "${PROJECT_BINARY_DIR}/libmscclpp_nccl.a"
    )
    add_library(mscclpp_nccl STATIC IMPORTED)
    set_target_properties(mscclpp_nccl PROPERTIES IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/libmscclpp_nccl.a)

endif()
