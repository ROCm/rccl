#!/bin/bash
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.

# #################################################
# helper functions
# #################################################
function display_help()
{
    echo "RCCL build & installation helper script"
    echo "./install [-h|--help] "
    echo "    [-h|--help] prints this help message."
    echo "    [-i|--install] install RCCL library (see --prefix argument below.)"
    echo "    [-d|--dependencies] install RCCL depdencencies."
    echo "    [-p|--package_build] Build RCCL package."
    echo "    [-t|--tests_build] Build unit tests, but do not run."
    echo "    [-r|--run_tests_quick] Run small subset of unit tests (must be built already.)"
    echo "    [-s|--static] Build RCCL as a static library instead of shared library."
    echo "    [--run_tests_all] Run all unit tests (must be built already.)"
    echo "    [--hcc] Build library using deprecated hcc compiler (default:hip-clang)."
    echo "    [--prefix] Specify custom directory to install RCCL to (default: /opt/rocm)."
    echo "    [--address-sanitizer] Build with address sanitizer enabled"
    echo "    [--build_allreduce_only] Build only AllReduce + sum + float kernel"
    echo "    [--rm-legacy-include-dir] Remove legacy include dir Packaging added for file/folder reorg backward compatibility"
}

# #################################################
# global variables
# #################################################
build_package=false
ROCM_PATH=${ROCM_PATH:="/opt/rocm"}
build_tests=false
run_tests=false
run_tests_all=false
build_release=true
build_address_sanitizer=false
install_library=false
build_hip_clang=true
clean_build=true
install_dependencies=false
build_static=false
build_allreduce_only=false
build_freorg_bkwdcomp=true

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,dependencies,package_build,tests_build,run_tests_quick,static,run_tests_all,hcc,hip-clang,no_clean,prefix:,address-sanitizer,build_allreduce_only,rm-legacy-include-dir --options hidptrs -- "$@")
else
    echo "Need a new version of getopt"
    exit 1
fi

if [[ $? -ne 0 ]]; then
    echo "getopt invocation failed; could not parse the command line";
    exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
    case "${1}" in
    -h|--help)
        display_help
        exit 0
        ;;
    -i|--install)
        install_library=true
        shift ;;
    -d|--dependencies)
	install_dependencies=true
	shift;;
    -p|--package_build)
        build_package=true
        shift ;;
    -t|--tests_build)
        build_tests=true
        shift ;;
    -r|--run_tests_quick)
        run_tests=true
        shift ;;
    -s|--static)
        build_static=true
        shift ;;
    --run_tests_all)
        run_tests=true
        run_tests_all=true
        shift ;;
    --hcc)
        build_hip_clang=false
        shift ;;
    --hip-clang)
        build_hip_clang=true
        shift ;;
    --no_clean)
        clean_build=false
        shift ;;
    --address-sanitizer)
        build_address_sanitizer=true
        shift ;;
    --build_allreduce_only)
        build_allreduce_only=true
        shift ;;
    --rm-legacy-include-dir)
        build_freorg_bkwdcomp=false
        shift ;;
    --prefix)
        install_prefix=${2}
        shift 2 ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
    esac
    done

ROCM_BIN_PATH=$ROCM_PATH/bin

# /etc/*-release files describe the system
if [[ -e "/etc/os-release" ]]; then
    source /etc/os-release
elif [[ -e "/etc/centos-release" ]]; then
    OS_ID=$(cat /etc/centos-release | awk '{print tolower($1)}')
    VERSION_ID=$(cat /etc/centos-release | grep -oP '(?<=release )[^ ]*' | cut -d "." -f1)
else
    echo "This script depends on the /etc/*-release files"
    exit 2
fi

# throw error code after running a command in the install script
check_exit_code( )
{
  if (( $1 != 0 )); then
    exit $1
  fi
}

if [[ "$build_release" == true ]]; then
    unit_test_path="./build/release/test/UnitTests"
else
    unit_test_path="./build/debug/test/UnitTests"
fi

if ($run_tests) && [[ -f $unit_test_path ]]; then
    if [[ "$build_tests" == false ]]; then
        clean_build=false
    fi
fi

# #################################################
# prep
# #################################################
# ensure a clean build environment
if ($clean_build); then
    if [[ "${build_release}" == true ]]; then
        rm -rf build/release
    else
        rm -rf build/debug
    fi
fi

# Create and go to the build directory.
mkdir -p build; cd build

if ($build_release); then
    mkdir -p release; cd release
else
    mkdir -p debug; cd debug
fi

# build type
if [[ "${build_release}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"
else
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Debug"
fi

# shared vs static
if [[ "${build_static}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_STATIC=ON"
fi

# sanitizer
if [[ "${build_address_sanitizer}" == true ]]; then
cmake_common_options="${cmake_common_options} -DBUILD_ADDRESS_SANITIZER=ON"
fi

#Enable backward compatibility wrappers
if [[ "${build_freorg_bkwdcomp}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=ON"
else
    cmake_common_options="${cmake_common_options} -DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=OFF"
fi

compiler=hipcc
if [[ "${build_hip_clang}" == false ]]; then
    compiler=hcc
fi

cmake_executable=cmake
case "${OS_ID}" in
    centos|rhel)
	cmake_executable=cmake3
	;;
    esac

if ($install_dependencies); then
    cmake_common_options="${cmake_common_options} -DINSTALL_DEPENDENCIES=ON"
fi

if ($build_allreduce_only); then
    cmake_common_options="${cmake_common_options} -DBUILD_ALLREDUCE_ONLY=ON"
fi
check_exit_code "$?"

if ($build_tests) || (($run_tests) && [[ ! -f ./test/UnitTests ]]); then
    CXX=$ROCM_BIN_PATH/$compiler $cmake_executable $cmake_common_options -DBUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX=$ROCM_PATH -DROCM_PATH=$ROCM_PATH ../../.
else
    CXX=$ROCM_BIN_PATH/$compiler $cmake_executable $cmake_common_options -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$ROCM_PATH -DROCM_PATH=$ROCM_PATH ../../.
fi
check_exit_code "$?"

if ($install_library); then
    make -j$(nproc) install
else
    make -j$(nproc)
fi
check_exit_code "$?"

if ($build_package); then
    make package
    check_exit_code "$?"
fi

# Optionally, run tests if they're enabled.
if ($run_tests); then
    if (test -f "./test/UnitTests"); then
        if ($run_tests_all); then
            ./test/UnitTests
        else
            ./test/UnitTests --gtest_filter="AllReduce.*"
        fi
    else
        echo "Unit tests have not been built yet; please re-run script with -t to build unit tests."
        exit 1
    fi
fi
