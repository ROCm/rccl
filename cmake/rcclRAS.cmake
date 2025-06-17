# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

cmake_minimum_required(VERSION 3.16)

message("Building rccl RAS client executable")

set(CMAKE_IGNORE_PATH "${ROCM_PATH}/lib" "${ROCM_PATH}/include")

add_executable(rcclras "${PROJECT_BINARY_DIR}/hipify/src/ras/client.cc")

target_include_directories(rcclras PRIVATE ${PROJECT_BINARY_DIR}/include)
target_include_directories(rcclras PRIVATE ${HIPIFY_DIR}/src)
target_include_directories(rcclras PRIVATE ${HIPIFY_DIR}/src/include)

if(BUILD_SHARED_LIBS)
  target_link_libraries(rcclras PRIVATE rccl)
  set_property(TARGET rcclras PROPERTY INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
else()
  add_dependencies(rccl-UnitTests rccl)
  target_link_libraries(rcclras PRIVATE dl rt -lrccl -L${CMAKE_BINARY_DIR} -lamdhip64 -L${ROCM_PATH}/lib)
endif()

set_target_properties(rcclras PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}")

rocm_install(TARGETS rcclras)
