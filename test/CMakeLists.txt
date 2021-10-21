# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
cmake_minimum_required(VERSION 2.8.12)

if(BUILD_TESTS)

  message("Going to build unit tests (Installed in /test/UnitTests)")

  include_directories(${GTEST_INCLUDE_DIRS})

  if(BUILD_ALLREDUCE_ONLY)
    set(TEST_SOURCES_SINGLE_PROCESS
      test_AllReduce.cpp
      test_AllReduceAbort.cpp
      test_AllReduceGroup.cpp
    )
  else()
    # Collect source files for tests
    set(TEST_SOURCES_SINGLE_PROCESS
      test_AllGather.cpp
      test_AllReduce.cpp
      test_AllReduceGroup.cpp
      test_Broadcast.cpp
      test_Reduce.cpp
      test_ReduceScatter.cpp
      test_GroupCalls.cpp
      test_CombinedCalls.cpp
      test_AllReduceAbort.cpp
      test_BroadcastAbort.cpp
      test_Scatter.cpp
      test_Gather.cpp
      test_AllToAll.cpp
      test_AllToAllv.cpp
    )
  endif()
  if(BUILD_ALLREDUCE_ONLY)
    set(TEST_SOURCES_MULTI_PROCESS
      test_AllReduceMultiProcess.cpp
      test_AllReduceGroupMultiProcess.cpp
    )
  else()
    set(TEST_SOURCES_MULTI_PROCESS
      test_AllGatherMultiProcess.cpp
      test_AllReduceMultiProcess.cpp
      test_AllReduceGroupMultiProcess.cpp
      test_AllToAllMultiProcess.cpp
      test_BroadcastMultiProcess.cpp
      test_CombinedCallsMultiProcess.cpp
      test_GatherMultiProcess.cpp
      test_GroupCallsMultiProcess.cpp
      test_ReduceMultiProcess.cpp
      test_ReduceScatterMultiProcess.cpp
      test_ScatterMultiProcess.cpp
    )
  endif()
  add_executable(UnitTests ${TEST_SOURCES_SINGLE_PROCESS})
  target_include_directories(UnitTests PRIVATE ${ROCM_PATH} ${GTEST_INCLUDE_DIRS})
  target_link_libraries(UnitTests PRIVATE ${GTEST_BOTH_LIBRARIES})
  target_link_libraries(UnitTests PRIVATE hip::host hip::device)

  add_executable(UnitTestsMultiProcess ${TEST_SOURCES_MULTI_PROCESS})
  target_include_directories(UnitTestsMultiProcess PRIVATE ${ROCM_PATH} ${GTEST_INCLUDE_DIRS})
  target_link_libraries(UnitTestsMultiProcess PRIVATE ${GTEST_BOTH_LIBRARIES})
  target_link_libraries(UnitTestsMultiProcess PRIVATE hip::host hip::device)

  # UnitTests using static library of rccl requires passing rccl
  # through -l and -L instead of command line input.
  if(BUILD_STATIC)
    add_dependencies(UnitTests rccl)
    target_link_libraries(UnitTests PRIVATE dl rt numa -lrccl -L${CMAKE_BINARY_DIR} -lrocm_smi64 -L${ROCM_PATH}/rocm_smi/lib)
    add_dependencies(UnitTestsMultiProcess rccl)
    target_link_libraries(UnitTestsMultiProcess PRIVATE dl rt numa -lrccl -L${CMAKE_BINARY_DIR} -lrocm_smi64 -L${ROCM_PATH}/rocm_smi/lib)
  else()
    target_link_libraries(UnitTests PRIVATE rccl)
    target_link_libraries(UnitTestsMultiProcess PRIVATE rt rccl)
  endif()
else()
  message("Not building unit tests")
endif()
