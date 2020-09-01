# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
cmake_minimum_required(VERSION 2.8.12)

if(BUILD_TESTS)

  message("Going to build unit tests (Installed in /test/UnitTests)")

  include_directories(${GTEST_INCLUDE_DIRS})  
  
  # Collect source files for tests
  set(TEST_SOURCES
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

  add_executable(UnitTests ${TEST_SOURCES})
  target_include_directories(UnitTests PRIVATE /opt/rocm ${GTEST_INCLUDE_DIRS})
  target_link_libraries(UnitTests PRIVATE ${GTEST_BOTH_LIBRARIES}  PRIVATE rccl)
else()
  message("Not building unit tests")
endif()
