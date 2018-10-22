cmake_minimum_required(VERSION 2.8.12)

project(rccl_gtests)

find_package(ROCM REQUIRED PATHS /opt/rocm)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip /opt/rocm/hcc)

find_package(hip REQUIRED)

# Clone Google Test
set(GOOGLETEST_DIR ${CMAKE_SOURCE_DIR}/lib/googletest CACHE PATH "googletest directory")
if(NOT EXISTS ${GOOGLETEST_DIR})
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} clone https://github.com/google/googletest ${GOOGLETEST_DIR}
    )
    add_subdirectory(lib/googletest)
endif()

enable_testing()
add_subdirectory(unit)
