# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
cmake_minimum_required(VERSION 2.8.12)

if(BUILD_TESTS)

  message("Going to build unit tests (Installed in /test/UnitTests)")

  include_directories(${GTEST_INCLUDE_DIRS})

  # Collect source files for tests
  set(TEST_SOURCES_SINGLE_PROCESS
    test_AllGather.cpp
    test_AllReduce.cpp
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

  set(TEST_SOURCES_MULTI_PROCESS
    test_AllGatherMultiProcess.cpp
    test_AllReduceMultiProcess.cpp
    test_AllToAllMultiProcess.cpp
    test_GatherMultiProcess.cpp
    test_ReduceMultiProcess.cpp
    test_ReduceScatterMultiProcess.cpp
    test_ScatterMultiProcess.cpp
  )

  add_executable(UnitTests ${TEST_SOURCES_SINGLE_PROCESS})
  target_include_directories(UnitTests PRIVATE /opt/rocm ${GTEST_INCLUDE_DIRS})
  target_link_libraries(UnitTests PRIVATE ${GTEST_BOTH_LIBRARIES})

  add_executable(UnitTestsMultiProcess ${TEST_SOURCES_MULTI_PROCESS})
  target_include_directories(UnitTestsMultiProcess PRIVATE /opt/rocm ${GTEST_INCLUDE_DIRS})
  target_link_libraries(UnitTestsMultiProcess PRIVATE ${GTEST_BOTH_LIBRARIES})

  # UnitTests using static library of rccl requires passing rccl
  # through -l and -L instead of command line input.
  if(BUILD_STATIC)
    add_dependencies(UnitTests rccl)
    target_link_libraries(UnitTests PRIVATE dl rt numa -lrccl -L${CMAKE_BINARY_DIR})
    target_link_libraries(UnitTests PRIVATE amdhip64 amd_comgr hsa-runtime64::hsa-runtime64)
    add_dependencies(UnitTestsMultiProcess rccl)
    target_link_libraries(UnitTestsMultiProcess PRIVATE dl rt numa -lrccl -L${CMAKE_BINARY_DIR})
    target_link_libraries(UnitTestsMultiProcess PRIVATE amdhip64 amd_comgr hsa-runtime64::hsa-runtime64)
  else()
    target_link_libraries(UnitTests PRIVATE rccl)
    target_link_libraries(UnitTestsMultiProcess PRIVATE rt rccl)
  endif()
else()
  message("Not building unit tests")
endif()
