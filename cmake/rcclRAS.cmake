# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

cmake_minimum_required(VERSION 3.16)

message("Building rccl RAS client executable")

add_executable(rcclras "${PROJECT_BINARY_DIR}/hipify/src/ras/client.cc")

target_include_directories(rcclras PRIVATE ${PROJECT_BINARY_DIR}/include)
target_include_directories(rcclras PRIVATE ${HIPIFY_DIR}/src)
target_include_directories(rcclras PRIVATE ${HIPIFY_DIR}/src/include)

target_link_libraries(rcclras PRIVATE hip::host)
target_link_libraries(rcclras PRIVATE dl)

if(BUILD_SHARED_LIBS)
  target_link_libraries(rcclras PRIVATE rccl)
  set_property(TARGET rcclras PROPERTY INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${CMAKE_BINARY_DIR};${ROCM_PATH}/lib")
else()
  add_dependencies(rcclras rccl)
  target_link_libraries(rcclras PRIVATE dl rt -lrccl -L${CMAKE_BINARY_DIR} -lamdhip64 -L${ROCM_PATH}/lib)
endif()

set_target_properties(rcclras PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR};${ROCM_PATH}/lib")

rocm_install(TARGETS rcclras)
