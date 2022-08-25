# Attempt to collect the latest git hash
execute_process(COMMAND git log --pretty=format:'%h' -n 1
                OUTPUT_VARIABLE GIT_REV
                ERROR_QUIET)

# Check if git information was found
if ("${GIT_REV}" STREQUAL "")
  set(GIT_VERSION "const char *rcclGitHash =\"Unknown \";")
else()
  # Check for changes (denote with a '+') after hash
  execute_process(
    COMMAND bash -c "git diff --quiet --exit-code || echo +"
    OUTPUT_VARIABLE GIT_DIFF)
  # Collect branch information
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    OUTPUT_VARIABLE GIT_BRANCH)

  string(STRIP "${GIT_REV}" GIT_REV)
  string(SUBSTRING "${GIT_REV}" 1 7 GIT_REV)
  string(STRIP "${GIT_DIFF}" GIT_DIFF)
  string(STRIP "${GIT_BRANCH}" GIT_BRANCH)

  set(GIT_VERSION "const char *rcclGitHash =\"${GIT_BRANCH}:${GIT_REV}${GIT_DIFF}\";")
endif()

# Compare file with older git version file (git_version.cpp)
if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp)
  #MESSAGE(STATUS "Found ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp")
  file(READ ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp PREV_GIT_VERSION)
  #message(STATUS "CURR GIT version: ${GIT_VERSION}")
  #message(STATUS "PREV GIT version: ${PREV_GIT_VERSION}")
  if (NOT "${GIT_VERSION}" STREQUAL "${PREV_GIT_VERSION}")
    message(STATUS "Updating git_version.cpp")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp "${GIT_VERSION}")
  else()
    message(STATUS "No changes to git_version.cpp required")
  endif()
else()
  # Create git_version.cpp if it doesn't exist yet
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp "${GIT_VERSION}")
endif()
