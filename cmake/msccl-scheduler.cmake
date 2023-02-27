# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Download msccl-scheduler
download_project(PROJ                msccl-scheduler
                 GIT_REPOSITORY      https://github.com/microsoft/msccl-scheduler.git
                 GIT_TAG             v0.1.0
                 SOURCE_DIR          ${CMAKE_CURRENT_BINARY_DIR}/msccl-scheduler-src
                 CONFIGURE_COMMAND   ""
                 BUILD_COMMAND       ""
                 INSTALL_COMMAND     ""
                 TEST_COMMAND        ""
                 UPDATE_DISCONNECTED TRUE
)

# Build msccl-scheduler
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env RCCL_BIN_HOME=${CMAKE_CURRENT_BINARY_DIR} RCCL_SRC_HOME=${CMAKE_CURRENT_SOURCE_DIR} make
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/msccl-scheduler-src/rccl)

# Install
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/msccl-scheduler-src/rccl/libmsccl-scheduler.so
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/msccl-scheduler-src/rccl/algorithms/
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rccl/msccl-algorithms)
