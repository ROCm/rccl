# Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")

project(rccl CXX)

find_package(ROCM
             REQUIRED
             PATHS
             /opt/rocm)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)

option(BUILD_TESTS "Build test programs" OFF)

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
rocm_setup_version(VERSION "2.7.0")

list(APPEND CMAKE_PREFIX_PATH
            /opt/rocm
            /opt/rocm/hip
            /opt/rocm/hcc)

find_package(hip REQUIRED)
message(STATUS "HIP compiler: ${HIP_COMPILER}")
message(STATUS "HIP runtime: ${HIP_RUNTIME}")

option(BUILD_SHARED_LIBS "Build as a shared library" ON)

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
    src/collectives/all_reduce.cc
    src/collectives/all_gather.cc
    src/collectives/reduce.cc
    src/collectives/broadcast.cc
    src/collectives/reduce_scatter.cc
    src/channel.cc
    src/misc/trees.cc
    src/misc/rings.cc
    src/misc/argcheck.cc
    src/misc/group.cc
    src/misc/utils.cc
    src/misc/ibvwrap.cc
    src/misc/nvmlwrap_stub.cc
    src/misc/topo.cc
    src/transport/net.cc
    src/transport/net_ib.cc
    src/transport/net_socket.cc
    src/transport/p2p.cc
    src/transport/shm.cc
    src/transport.cc
    src/bootstrap.cc
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

target_link_libraries(rccl
  PRIVATE --amdgpu-target=gfx803
  PRIVATE --amdgpu-target=gfx900
  PRIVATE --amdgpu-target=gfx906
  PRIVATE --amdgpu-target=gfx908)

if("${HIP_COMPILER}" MATCHES "clang")
  target_compile_options(rccl
    PRIVATE --amdgpu-target=gfx803
    PRIVATE --amdgpu-target=gfx900
    PRIVATE --amdgpu-target=gfx906
    PRIVATE --amdgpu-target=gfx908
    PRIVATE -fgpu-rdc)
  target_link_libraries(rccl PRIVATE -fgpu-rdc)
  target_include_directories(rccl PRIVATE /opt/rocm/hsa/include)
endif()

if("${HIP_COMPILER}" MATCHES "hcc")
  target_link_libraries(rccl PRIVATE -hc-function-calls)
endif()

if(TARGET hip::device)
  target_link_libraries(rccl PRIVATE hip::device)
  target_link_libraries(rccl INTERFACE hip::host)
else()
  target_link_libraries(rccl PUBLIC hip::hip_hcc ${hcc_LIBRARIES} numa)
endif()

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

set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc")
set(CPACK_RPM_PACKAGE_REQUIRES "hip_hcc")

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/opt" "/opt/rocm")

rocm_create_package(
  NAME
  rccl
  DESCRIPTION
  "Optimized primitives for collective multi-GPU communication"
  MAINTAINER
  "<no-reply@amd.com>"
  LDCONFIG)

rocm_install_symlink_subdir(rccl)

if(BUILD_TESTS)
  add_subdirectory(test)
endif()
