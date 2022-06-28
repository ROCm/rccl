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

# Compare file with existing file (if any)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/git_version.cpp)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/git_version.cpp GIT_VERSION_)
else()
  set(GIT_VERSION_ "")
endif()

# Write updated file
if (NOT "${GIT_VERSION}" STREQUAL "${GIT_VERSION_}")
  file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/git_version.cpp "${GIT_VERSION}")
endif()
