# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
cmake_minimum_required(VERSION 2.8.12)

if(BUILD_TESTS)

  message("Going to build unit tests (Installed in /test/UnitTests)")

  # chrpath is required to properly set rpath for the UnitTests executable
  find_program(CHRPATH chrpath)
  if(NOT CHRPATH)
      message(FATAL_ERROR "chrpath is required for UnitTests. Please install (e.g. sudo apt-get install chrpath)")
  endif()

  find_package(GTest REQUIRED)
  if (NOT GTest_FOUND)
    message(FATAL_ERROR "Please ensure GTest is installed on the system")
  endif()

  include_directories(${GTEST_INCLUDE_DIR})  

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
  )

  add_executable(UnitTests ${TEST_SOURCES})
  target_include_directories(UnitTests PRIVATE /opt/rocm)
  target_link_libraries(UnitTests PRIVATE ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} PRIVATE rccl)
else()
  message("Not building unit tests")
endif()
