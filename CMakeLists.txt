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

if(NOT DEFINED ROCR_PATH)
    if(NOT DEFINED ENV{ROCR_PATH})
        set(ROCR_PATH "/opt/rocm/hsa" CACHE PATH "Path to which ROCR has been installed")
    else()
        set(ROCR_PATH $ENV{ROCR_PATH} CACHE PATH "Path to which ROCR has been installed")
    endif()
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND CMAKE_INSTALL_PREFIX MATCHES "/usr/local")
    set(CMAKE_INSTALL_PREFIX "/opt/rocm/rccl" CACHE PATH "RCCL will be installed at: " FORCE)
endif()

message(STATUS "RCCL will be installed at: " ${CMAKE_INSTALL_PREFIX})

set (CMAKE_CXX_COMPILER "${HCC_HOME}/bin/hcc")
set (CMAKE_C_COMPILER "${HCC_HOME}/bin/hcc")

include_directories(${PROJECT_SOURCE_DIR}/inc)
include_directories(${HIP_PATH}/include)

set(RCCL_COMPILE_FLAGS "-hc -I${ROCR_PATH}/include")
set (RCCL_BUILD_TARGETS "--amdgpu-target=gfx803 --amdgpu-target=gfx900")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${RCCL_COMPILE_FLAGS}")

execute_process(COMMAND ${HCC_HOME}/bin/hcc-config --shared --cxxflags --ldflags OUTPUT_VARIABLE RCCL_SHARED_FLAGS)

execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE HCC_VERSION)

message(STATUS "HCC Version: ${HCC_VERSION}")

set (RCCL_SRC src/rccl.cpp)

set(RCCL_LINK_LIBRARIES "-L${HIP_PATH}/lib -lhip_device -lhip_hcc ${RCCL_BUILD_TARGETS} ${RCCL_SHARED_FLAGS}")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${RCCL_LINK_LIBRARIES}")

add_library(${RCCL} SHARED ${RCCL_SRC})

install(TARGETS ${RCCL} DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/inc/rccl.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/rcclCheck.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

