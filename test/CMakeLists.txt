# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.

cmake_minimum_required(VERSION 2.8.12)

if(BUILD_TESTS)

  message("Building rccl unit tests (Installed in /test/rccl-UnitTests)")

  find_package(hsa-runtime64 PATHS /opt/rocm )
  if(${hsa-runtime64_FOUND})
    message("hsa-runtime64 found @  ${hsa-runtime64_DIR} ")
  else()
    message("find_package did NOT find hsa-runtime64, finding it the OLD Way")
    message("Looking for header files in ${ROCR_INC_DIR}")
    message("Looking for library files in ${ROCR_LIB_DIR}")

    # Search for ROCr header file in user defined locations
    find_path(ROCR_HDR hsa/hsa.h PATHS ${ROCR_INC_DIR} "/opt/rocm" PATH_SUFFIXES include REQUIRED)
    INCLUDE_DIRECTORIES(${ROCR_HDR})

    # Search for ROCr library file in user defined locations
    find_library(ROCR_LIB ${CORE_RUNTIME_TARGET} PATHS ${ROCR_LIB_DIR} "/opt/rocm" PATH_SUFFIXES lib lib64 REQUIRED)
  endif()

  include_directories(${GTEST_INCLUDE_DIRS} ./common)

  # Collect testing framework source files
  set (COMMON_SOURCE_FILES
    common/main.cpp
    common/CollectiveArgs.cpp
    common/EnvVars.cpp
    common/PrepDataFuncs.cpp
    common/PtrUnion.cpp
    common/TestBed.cpp
    common/TestBedChild.cpp
    )

  # Collect source files for tests
  if(BUILD_ALLREDUCE_ONLY)
    set(TEST_SOURCE_FILES
      AllReduce_Tests.cpp
    )
  else()
    set(TEST_SOURCE_FILES
      AllGatherTests.cpp
      AllReduceTests.cpp
      AllToAllTests.cpp
      AllToAllVTests.cpp
      BroadcastTests.cpp
      GatherTests.cpp
      GroupCallTests.cpp
      NonBlockingTests.cpp
      ReduceScatterTests.cpp
      ReduceTests.cpp
      ScatterTests.cpp
      SendRecvTests.cpp
      StandaloneTests.cpp
      )
  endif()

  add_executable(rccl-UnitTests ${COMMON_SOURCE_FILES} ${TEST_SOURCE_FILES})
  target_include_directories(rccl-UnitTests PRIVATE ${ROCM_PATH} ${GTEST_INCLUDE_DIRS})
  target_include_directories(rccl-UnitTests PRIVATE ${PROJECT_BINARY_DIR}/include)        # for generated rccl.h header
  target_include_directories(rccl-UnitTests PRIVATE ${PROJECT_BINARY_DIR}/hipify/src/include) # for rccl_bfloat16.h
  target_link_libraries(rccl-UnitTests PRIVATE ${GTEST_BOTH_LIBRARIES})
  target_link_libraries(rccl-UnitTests PRIVATE hip::host hip::device hsa-runtime64::hsa-runtime64)
  target_link_libraries(rccl-UnitTests PRIVATE Threads::Threads)

  # rccl-UnitTests using static library of rccl requires passing rccl
  # through -l and -L instead of command line input.
  if(BUILD_STATIC)
    add_dependencies(rccl-UnitTests rccl)
    target_link_libraries(rccl-UnitTests PRIVATE dl rt numa -lrccl -L${CMAKE_BINARY_DIR} -lrocm_smi64 -L${ROCM_PATH}/lib -L${ROCM_PATH}/rocm_smi/lib)
  else()
    target_link_libraries(rccl-UnitTests PRIVATE rccl)
  endif()
  set_property(TARGET rccl-UnitTests PROPERTY INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${ROCM_PATH}/lib;${CMAKE_BINARY_DIR}")
  set_property(TARGET rccl-UnitTests PROPERTY BUILD_RPATH "${CMAKE_BINARY_DIR};${ROCM_PATH}/lib")
  rocm_install(TARGETS rccl-UnitTests COMPONENT tests)
else()
  message("Not building rccl unit tests")
endif()
