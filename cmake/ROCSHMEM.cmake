# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

include(ExternalProject)

function(add_rocshmem_targets)
    if(ROCSHMEM_INSTALL_DIR)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
        find_package(rocshmem_static)
    endif()

    if(NOT rocshmem_static_FOUND)
        set(ROCSHMEM_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/rocshmem)
        set(ROCSHMEM_INCLUDE_DIRS "${ROCSHMEM_INSTALL_DIR}/include")
        set(ROCSHMEM_STATIC_LIB "${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a")
        execute_process(
            COMMAND mkdir -p $(ROCSHMEM_INSTALL_DIR)
        )

        set(EXT_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ext-src)

        if (NOT ROCSHMEM_SOURCE)
            add_custom_command(
                OUTPUT
                    ${EXT_SOURCE}/rocSHMEM/CMakeLists.txt
                COMMAND git submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Checking out submodules for rocSHMEM"
            )
            add_custom_target(
                checkout_submodules
                DEPENDS
                    ${EXT_SOURCE}/rocSHMEM/CMakeLists.txt
            )
        endif()
    endif()
endfunction()
