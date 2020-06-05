#!/bin/bash
# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

# #################################################
# helper functions
# #################################################
function display_help()
{
    echo "RCCL build & installation helper script"
    echo "./install [-h|--help] "
    echo "    [-h|--help] prints this help message."
    echo "    [-i|--install] install RCCL library (see --prefix argument below.)"
    echo "    [-p|--package_build] Build RCCL package."
    echo "    [-t|--tests_build] Build unit tests, but do not run."
    echo "    [-r|--run_tests] Run unit tests (must be built already.)"
    echo "    [--hcc] Build library using deprecated hcc compiler (default:hip-clang)."
    echo "    [--prefix] Specify custom directory to install RCCL to (default: /opt/rocm)."
}

# #################################################
# global variables
# #################################################
default_path=/opt/rocm
build_package=false
install_prefix=$default_path
build_tests=false
run_tests=false
build_release=true
install_library=false
build_hip_clang=true
install_dependencies=false
# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,package_build,tests_build,run_tests,hcc,hip-clang,prefix: --options hiptr -- "$@")
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
    -r|--run_tests)
        run_tests=true
        shift ;;
    --hcc)
        build_hip_clang=false
        shift ;;
    --hip-clang)
        build_hip_clang=true
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

rocm_path=/opt/rocm/bin

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

# #################################################
# prep
# #################################################
# ensure a clean build environment
if [[ "${build_release}" == true ]]; then
    rm -rf build/release
else
    rm -rf build/debug
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
    if [[ -e /etc/redhat-release ]]; then
	yum install chrpath libgomp
    else
	apt install chrpath libomp-dev
    fi
fi
check_exit_code "$?"

if ($build_tests); then
    CXX=$rocm_path/$compiler $cmake_executable $cmake_common_options -DBUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX=$install_prefix ../../.
else
    CXX=$rocm_path/$compiler $cmake_executable $cmake_common_options -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$install_prefix ../../.
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
        HSA_FORCE_FINE_GRAIN_PCIE=1 ./test/UnitTests
    else
        echo "Unit tests have not been built yet; please re-run script with -t to build unit tests."
        exit 1
    fi
fi
