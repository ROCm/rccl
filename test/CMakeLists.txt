# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
cmake_minimum_required(VERSION 2.8.12)

if(BUILD_TESTS)

  message("Building unit tests (Installed in /test/UnitTests)")

  find_program(CHRPATH chrpath)
  if(NOT CHRPATH)
      message(FATAL_ERROR "chrpath is required for UnitTests. Please install (e.g. sudo apt-get install chrpath)")
  endif()

  find_package(hsa-runtime64 PATHS /opt/rocm )
  if(${hsa-runtime64_FOUND})
    message("hsa-runtime64 found @  ${hsa-runtime64_DIR} ")
  else()
    message("find_package did NOT find hsa-runtime64, finding it the OLD Way")
    message("Looking for header files in ${ROCR_INC_DIR}")
    message("Looking for library files in ${ROCR_LIB_DIR}")

    # Search for ROCr header file in user defined locations
    find_path(ROCR_HDR hsa.h PATHS ${ROCR_INC_DIR} "/opt/rocm" PATH_SUFFIXES include/hsa REQUIRED)
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
      AllReduce_Clique.cpp
      AllReduce_GroupCall.cpp
      AllReduce_InPlace.cpp
      AllReduce_ManagedMem.cpp
      AllReduce_OutOfPlace.cpp
      AllReduce_PreMultScalar.cpp
    )
  else()
    set(TEST_SOURCE_FILES
      #AllReduce
      AllReduce_Clique.cpp
      AllReduce_GroupCall.cpp
      AllReduce_InPlace.cpp
      AllReduce_ManagedMem.cpp
      AllReduce_OutOfPlace.cpp
      AllReduce_PreMultScalar.cpp
      #AllGather
      AllGather_InPlace.cpp
      AllGather_ManagedMem.cpp
      AllGather_OutOfPlace.cpp
      #AllToAll
      AllToAll_OutOfPlace.cpp
      AllToAll_ManagedMem.cpp
      #AllToAllv
      AllToAllv_OutOfPlace.cpp
      #Broadcast
      Broadcast_InPlace.cpp
      Broadcast_ManagedMem.cpp
      Broadcast_OutOfPlace.cpp
      #Reduce
      Reduce_InPlace.cpp
      Reduce_ManagedMem.cpp
      Reduce_OutOfPlace.cpp
      #ReduceScatter
      ReduceScatter_InPlace.cpp
      ReduceScatter_ManagedMem.cpp
      ReduceScatter_OutOfPlace.cpp
      #Scatter
      Scatter_InPlace.cpp
      Scatter_ManagedMem.cpp
      Scatter_OutOfPlace.cpp
      #Gather
      Gather_InPlace.cpp
      Gather_ManagedMem.cpp
      Gather_OutOfPlace.cpp
      #SendRecv
      SendRecv_SinglePairs.cpp
      )
  endif()

  add_executable(UnitTests ${COMMON_SOURCE_FILES} ${TEST_SOURCE_FILES})
  target_include_directories(UnitTests PRIVATE ${ROCM_PATH} ${GTEST_INCLUDE_DIRS})
  target_link_libraries(UnitTests PRIVATE ${GTEST_BOTH_LIBRARIES})
  target_link_libraries(UnitTests PRIVATE hip::host hip::device hsa-runtime64::hsa-runtime64)

  # UnitTests using static library of rccl requires passing rccl
  # through -l and -L instead of command line input.
  if(BUILD_STATIC)
    add_dependencies(UnitTests rccl)
    target_link_libraries(UnitTests PRIVATE dl rt numa -lrccl -L${CMAKE_BINARY_DIR} -lrocm_smi64 -L${ROCM_PATH}/rocm_smi/lib)
  else()
    target_link_libraries(UnitTests PRIVATE rccl)
  endif()
  # HIPCC adds /opt/rocm/lib as RPATH, even though the install process is supposed to
  # remove RPATH.  It also occurs before any user-specified rpath, which effectively overrides the user rpath.
  #  As a work-around, set the correct RPATH for the unit test executable as a post-install step
  # if (CMAKE_INSTALL_PREFIX MATCHES "${ROCM_PATH}")
  #   # install_prefix/CMAKE_INSTALL_PREFIX was not explicitly specified, so look in build/release
  #   add_custom_command( TARGET UnitTests POST_BUILD COMMAND chrpath ARGS -r ${CMAKE_BINARY_DIR}:${ROCM_PATH}/lib ${CMAKE_BINARY_DIR}/test/UnitTests)
  # else()
  #   add_custom_command( TARGET UnitTests POST_BUILD COMMAND chrpath ARGS -r ${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib ${CMAKE_INSTALL_PREFIX}/test/UnitTests)
  # endif()
  set_property(TARGET UnitTests PROPERTY INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${ROCM_PATH}/lib;${CMAKE_BINARY_DIR}")
  set_property(TARGET UnitTests PROPERTY BUILD_RPATH "${CMAKE_BINARY_DIR};${ROCM_PATH}/lib")
  rocm_install(TARGETS UnitTests COMPONENT tests)
else()
  message("Not building unit tests")
endif()
