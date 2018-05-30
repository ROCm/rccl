cmake_minimum_required(VERSION 3.5.0)

set ( RCCL "rccl")
set ( RCCL_PACKAGE "${RCCL}")
set ( RCCL_COMPONENT "lib${RCCL}")
project ( ${RCCL} )

set ( BUILD_VERSION_MAJOR ${VERSION_MAJOR} )
set ( LIB_VERSION_STRING "0.1.1")

if(NOT DEFINED HCC_HOME)
    if(NOT DEFINED ENV{HCC_HOME})
        set(HCC_HOME "/opt/rocm/hcc" CACHE PATH "Path to which HCC has been installed")
    else()
        set(HCC_HOME $ENV{HCC_HOME} CACHE PATH "Path to which HCC has been installed")
    endif()
endif()

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

set (CMAKE_CXX_COMPILER "${HCC_HOME}/bin/hcc")
set (CMAKE_C_COMPILER "${HCC_HOME}/bin/hcc")

include_directories(${PROJECT_SOURCE_DIR}/inc)
include_directories(${HIP_PATH}/include)

set (RCCL_BUILD_TARGETS "--amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${RCCL_COMPILE_FLAGS}")

execute_process(COMMAND ${HCC_HOME}/bin/hcc-config --shared --cxxflags --ldflags OUTPUT_VARIABLE RCCL_SHARED_FLAGS)

execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE HCC_VERSION)

message(STATUS "HCC Version: ${HCC_VERSION}")

set (RCCL_SRC src/rccl.cpp)

set(RCCL_LINK_LIBRARIES "-L${HIP_PATH}/lib -lhip_device -lhip_hcc ${RCCL_BUILD_TARGETS}")

add_library(${RCCL} SHARED ${RCCL_SRC})
