# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
cmake_minimum_required(VERSION 2.8.12)

if(BUILD_TESTS)

  message("Going to build unit tests (Installed in /test/UnitTests)")

  find_program(CHRPATH chrpath)
  if(NOT CHRPATH)
      message(FATAL_ERROR "chrpath is required for UnitTests. Please install (e.g. sudo apt-get install chrpath)")
  endif()

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

  find_program( rocminfo_executable rocminfo )
  execute_process(COMMAND bash "-c" "${rocminfo_executable} | grep 'Device Type' | grep GPU | wc -l | tr -d '\n'" OUTPUT_VARIABLE gtest_num_gpus)
  if(${gtest_num_gpus} EQUAL "0" OR ${gtest_num_gpus} EQUAL "1")
    set(gtest_num_gpus "2")
  endif()
  target_compile_options(UnitTests PRIVATE -DGTESTS_NUM_GPUS=${gtest_num_gpus})

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
  # HIPCC adds /opt/rocm/lib as RPATH, even though the install process is supposed to
  # remove RPATH.  It also occurs before any user-specified rpath, which effectively overrides the user rpath.
  #  As a work-around, set the correct RPATH for the unit test executable as a post-install step
  if (CMAKE_INSTALL_PREFIX MATCHES "${ROCM_PATH}")
    # install_prefix/CMAKE_INSTALL_PREFIX was not explicitly specified, so look in build/release
    add_custom_command( TARGET UnitTests POST_BUILD COMMAND chrpath ARGS -r ${CMAKE_BINARY_DIR}:${ROCM_PATH}/lib ${CMAKE_BINARY_DIR}/test/UnitTests)
    add_custom_command( TARGET UnitTestsMultiProcess POST_BUILD COMMAND chrpath ARGS -r ${CMAKE_BINARY_DIR}:${ROCM_PATH}/lib ${CMAKE_BINARY_DIR}/test/UnitTestsMultiProcess)
  else()
    add_custom_command( TARGET UnitTests POST_BUILD COMMAND chrpath ARGS -r ${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib ${CMAKE_INSTALL_PREFIX}/test/UnitTests)
    add_custom_command( TARGET UnitTestsMultiProcess POST_BUILD COMMAND chrpath ARGS -r ${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib ${CMAKE_INSTALL_PREFIX}/test/UnitTestsMultiProcess)
  endif()
else()
  message("Not building unit tests")
endif()
