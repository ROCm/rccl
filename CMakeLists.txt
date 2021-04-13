# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.5)

# We use C++14 features, this will add compile option: -std=c++14
set( CMAKE_CXX_STANDARD 14 )
# Without this line, it will add -std=gnu++14 instead, which has some issues.
set( CMAKE_CXX_EXTENSIONS OFF )

set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")

project(rccl CXX)

# Detect compiler support for target ID
# This section is deprecated. Please use rocm_check_target_ids for future use.
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
        OUTPUT_VARIABLE CXX_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
endif()

#Set the AMDGPU_TARGETS with backward compatiblity
if(COMMAND rocm_check_target_ids)
   rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
       TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx1030"
   )
else()
    # Use target ID syntax if supported for AMDGPU_TARGETS
    if(TARGET_ID_SUPPORT)
        set(DEFAULT_AMDGPU_TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx1030")
    else()
        set(DEFAULT_AMDGPU_TARGETS "gfx803;gfx900;gfx906;gfx908")
    endif()
endif()
set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")

option(BUILD_TESTS "Build test programs" OFF)
option(INSTALL_DEPENDENCIES "Force install dependencies" OFF)

include(cmake/Dependencies.cmake)

# parse version from Makefile NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH must exist
# NCCL_SUFFIX is optional NCCL_VERSION formatting is ((X) * 1000 + (Y) * 100 +
# (Z)) so we must first detect one or two digits first
file(READ makefiles/version.mk version_mk_text)
if("${version_mk_text}" MATCHES "NCCL_MAJOR *:= *([0-9]*)")
  set(NCCL_MAJOR ${CMAKE_MATCH_1})
else()
  message(FATAL_ERROR "Failed to parse NCCL_MAJOR")
endif()
if("${version_mk_text}" MATCHES "NCCL_MINOR *:= *([0-9]*)")
  set(NCCL_MINOR ${CMAKE_MATCH_1})
else()
  message(FATAL_ERROR "Failed to parse NCCL_MINOR")
endif()
if("${version_mk_text}" MATCHES "NCCL_PATCH *:= *([0-9]*)")
  set(NCCL_PATCH ${CMAKE_MATCH_1})
else()
  message(FATAL_ERROR "Failed to parse NCCL_PATCH")
endif()
if("${version_mk_text}" MATCHES "NCCL_SUFFIX *:= *([0-9]*)")
  set(NCCL_SUFFIX ${CMAKE_MATCH_1})
else()
  set(NCCL_SUFFIX)
endif()
if("${version_mk_text}" MATCHES "PKG_REVISION *:= *([0-9]*)")
  set(PKG_REVISION ${CMAKE_MATCH_1})
else()
  message(FATAL_ERROR "Failed to parse PKG_REVISION")
endif()
if("${NCCL_PATCH}" MATCHES "[0-9][0-9]")
  set(NCCL_VERSION "${NCCL_MAJOR}${NCCL_MINOR}${NCCL_PATCH}")
else()
  set(NCCL_VERSION "${NCCL_MAJOR}${NCCL_MINOR}0${NCCL_PATCH}")
endif()

# Setup VERSION
set(VERSION_STRING "${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}")
rocm_setup_version(VERSION ${VERSION_STRING})

list(APPEND CMAKE_PREFIX_PATH
            /opt/rocm
            /opt/rocm/hip
            /opt/rocm/llvm
            /opt/rocm/hcc)

find_package(hip REQUIRED)
message(STATUS "HIP compiler: ${HIP_COMPILER}")
message(STATUS "HIP runtime: ${HIP_RUNTIME}")

if(BUILD_STATIC)
  option(BUILD_SHARED_LIBS "Build as a shared library" OFF)
else()
  option(BUILD_SHARED_LIBS "Build as a shared library" ON)
endif()

configure_file(src/nccl.h.in ${PROJECT_BINARY_DIR}/rccl.h)
configure_file(src/nccl.h.in ${PROJECT_BINARY_DIR}/nccl.h)

include_directories(${PROJECT_BINARY_DIR}) # for generated rccl.h header
include_directories(src)
include_directories(src/include)
include_directories(src/collectives)
include_directories(src/collectives/device)

set(CU_SOURCES
    src/collectives/device/all_reduce.cu
    src/collectives/device/all_gather.cu
    src/collectives/device/reduce.cu
    src/collectives/device/broadcast.cu
    src/collectives/device/reduce_scatter.cu
    src/collectives/device/sendrecv.cu
    src/collectives/device/functions.cu)

set(CPP_SOURCES)
foreach(filename ${CU_SOURCES})
  string(REPLACE ".cu"
                 ".cpp"
                 cpp_filename
                 ${filename})
  configure_file(${filename} ${cpp_filename} COPYONLY)
  list(APPEND CPP_SOURCES ${cpp_filename})
endforeach(filename)

set(CC_SOURCES
    src/init.cc
    src/graph/trees.cc
    src/graph/rings.cc
    src/graph/paths.cc
    src/graph/search.cc
    src/graph/connect.cc
    src/graph/tuning.cc
    src/graph/topo.cc
    src/graph/xml.cc
    src/graph/rome_models.cc
    src/collectives/all_reduce_api.cc
    src/collectives/all_gather_api.cc
    src/collectives/reduce_api.cc
    src/collectives/broadcast_api.cc
    src/collectives/reduce_scatter_api.cc
    src/collectives/sendrecv_api.cc
    src/collectives/gather_api.cc
    src/collectives/scatter_api.cc
    src/collectives/all_to_all_api.cc
    src/collectives/all_to_allv_api.cc
    src/channel.cc
    src/clique/CliqueManager.cc     # RCCL
    src/clique/HandleCache.cc       # RCCL
    src/clique/HandleShm.cc         # RCCL
    src/clique/Hash.cc              # RCCL
    src/clique/MsgQueue.cc          # RCCL
    src/clique/ShmObject.cc         # RCCL
    src/misc/argcheck.cc
    src/misc/nvmlwrap_stub.cc
    src/misc/utils.cc
    src/misc/ibvwrap.cc
    src/misc/nvmlwrap_stub.cc
    src/transport/coll_net.cc
    src/transport/net.cc
    src/transport/net_ib.cc
    src/transport/net_socket.cc
    src/transport/p2p.cc
    src/transport/shm.cc
    src/transport.cc
    src/debug.cc
    src/group.cc
    src/bootstrap.cc
    src/proxy.cc
    src/enqueue.cc)

foreach(filename ${CC_SOURCES})
  list(APPEND CPP_SOURCES ${filename})
endforeach(filename)

add_library(rccl ${CPP_SOURCES})

if(TRACE)
  add_definitions(-DENABLE_TRACE)
endif()

if(PROFILE)
  add_definitions(-DENABLE_PROFILING)
endif()

set(COLLTRACE 1 CACHE BOOL "Collective Trace Option")
if(COLLTRACE)
  add_definitions(-DENABLE_COLLTRACE)
endif()

foreach(target ${AMDGPU_TARGETS})
target_link_libraries(rccl PRIVATE --cuda-gpu-arch=${target})
endforeach()

if("${HIP_COMPILER}" MATCHES "clang")
  target_compile_options(rccl PRIVATE -fvisibility=hidden)
  foreach(target ${AMDGPU_TARGETS})
    target_compile_options(rccl PRIVATE --cuda-gpu-arch=${target} PRIVATE -fgpu-rdc)
  endforeach()
  target_link_libraries(rccl PRIVATE -fgpu-rdc)
  target_include_directories(rccl PRIVATE /opt/rocm/hsa/include)
  find_program( hipcc_executable hipcc )
  execute_process(COMMAND bash "-c" "${hipcc_executable} -help | grep 'parallel-jobs'" OUTPUT_VARIABLE hipcc_parallel_jobs)
  if("${hipcc_parallel_jobs}" MATCHES "parallel-jobs")
    target_compile_options(rccl PRIVATE -parallel-jobs=4 PRIVATE -Wno-format-nonliteral)
    target_link_libraries(rccl PRIVATE -parallel-jobs=4)
  endif()

  # RCCL static lib uses -fgpu-rdc which requires hipcc as the linker and archiver
  if(BUILD_STATIC)
    target_link_libraries(rccl PRIVATE --emit-static-lib)
    set(CMAKE_AR "${hipcc_executable}")
    get_property(link_libraries TARGET rccl PROPERTY LINK_LIBRARIES)
    string (REPLACE ";" " " LINK_PROPS "${link_libraries}")
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> -o <TARGET> ${LINK_PROPS} <LINK_FLAGS> <OBJECTS>")
  endif()
endif()

if("${HIP_COMPILER}" MATCHES "hcc")
  find_program( hcc_executable hcc )
  execute_process(COMMAND bash "-c" "${hcc_executable} --version | sed -e '1!d' -e 's/.*based on HCC\\s*//'" OUTPUT_VARIABLE hcc_version_string)
  execute_process(COMMAND bash "-c" "echo \"${hcc_version_string}\" | awk -F\".\" '{ printf $1}'" OUTPUT_VARIABLE hcc_major_version)
  execute_process(COMMAND bash "-c" "echo \"${hcc_version_string}\" | awk -F\".\" '{ printf $2}'" OUTPUT_VARIABLE hcc_minor_version)
  if ("${hcc_major_version}.${hcc_minor_version}" VERSION_LESS "4.0")
    target_link_libraries(rccl PRIVATE -hc-function-calls)
  endif()
endif()

target_link_libraries(rccl PRIVATE hip::device numa)
target_link_libraries(rccl INTERFACE hip::host)

#Setup librccl.so version
rocm_set_soversion(rccl "1.0")

rocm_install_targets(TARGETS
                     rccl
                     PREFIX
                     rccl)
install(FILES ${PROJECT_BINARY_DIR}/rccl.h
        DESTINATION rccl/${CMAKE_INSTALL_INCLUDEDIR})

rocm_export_targets(NAMESPACE
                    roc::
                    PREFIX
                    rccl
                    TARGETS
                    rccl
                    DEPENDS
                    hip)

set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip-rocclr (>= 3.5.0)")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_RPM_PACKAGE_REQUIRES "hip-rocclr >= 3.5.0")
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/opt" "/opt/rocm")

find_file (DEBIAN debian_version debconf.conf PATHS /etc)
if(DEBIAN)
  # Write copyright file
  file(WRITE "${CMAKE_BINARY_DIR}/copyright"
  "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: rccl
Source: https://github.com/ROCmSoftwarePlatform/rccl

Files: *
Copyright: (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
Modifications Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
License: See LICENSE.txt for license information\n")
  install(FILES "${CMAKE_BINARY_DIR}/copyright" DESTINATION /usr/share/doc/rccl)
  # Write changelog file
  find_program( date_executable date )
  execute_process(COMMAND ${date_executable} -R OUTPUT_VARIABLE TIMESTAMP)
  file(WRITE "${CMAKE_BINARY_DIR}/changelog"
  "rccl (${VERSION_STRING}-1) unstable; urgency=medium

  * Initial release.

 -- RCCL Maintainer <rccl-maintainer@amd.com>  ${TIMESTAMP}\n")
  find_program( gzip_executable gzip )
  execute_process(COMMAND bash "-c" "${gzip_executable} -9 -c ${CMAKE_BINARY_DIR}/changelog"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR} OUTPUT_FILE "${CMAKE_BINARY_DIR}/changelog.Debian.gz")
  install(FILES "${CMAKE_BINARY_DIR}/changelog.Debian.gz" DESTINATION /usr/share/doc/rccl)
  set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "ROCm Communication Collectives Library
  Optimized primitives for collective multi-GPU communication")
endif()

rocm_create_package(
  NAME
  rccl
  DESCRIPTION
  "ROCm Communication Collectives Library"
  MAINTAINER
  "RCCL Maintainer <rccl-maintainer@amd.com>"
  LDCONFIG)

rocm_install_symlink_subdir(rccl)

if(BUILD_TESTS)
  add_subdirectory(test)
endif()
