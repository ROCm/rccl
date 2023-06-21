#!/bin/bash
# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.

# #################################################
# helper functions
# #################################################
function display_help()
{
    echo "RCCL build & installation helper script"
    echo "./install [-h|--help] "
    echo "       --address-sanitizer     Build with address sanitizer enabled"
    echo "       --build_allreduce_only  Build only AllReduce + sum + float kernel"
    echo "    -d|--dependencies          Install RCCL depdencencies"
    echo "       --debug                 Build debug library"
    echo "       --disable_backtrace     Build without custom backtrace support"
    echo "       --disable-colltrace     Build without collective trace"
    echo "       --fast                  Quick-build RCCL (local gpu arch only, no backtrace, and collective trace support)"
    echo "    -h|--help                  Prints this help message"
    echo "    -i|--install               Install RCCL library (see --prefix argument below)"
    echo "    -l|--limit-nprocs          Limit the number of procs to 16 while building"
    echo "       --local_gpu_only        Only compile for local GPU architecture"
    echo "       --no_clean              Don't delete files if they already exist"
    echo "       --npkit-enable          Compile with npkit enabled"
    echo "    -p|--package_build         Build RCCL package"
    echo "       --prefix                Specify custom directory to install RCCL to (default: /opt/rocm)"
    echo "       --rm-legacy-include-dir Remove legacy include dir Packaging added for file/folder reorg backward compatibility"
    echo "       --run_tests_all         Run all rccl unit tests (must be built already)"
    echo "    -r|--run_tests_quick       Run small subset of rccl unit tests (must be built already)"
    echo "       --static                Build RCCL as a static library instead of shared library"
    echo "    -t|--tests_build           Build rccl unit tests, but do not run"
    echo "       --time-trace            Plot the build time of RCCL"
    echo "       --verbose               Show compile commands"
}

# #################################################
# global variables
# #################################################
ROCM_PATH=${ROCM_PATH:="/opt/rocm"}

build_address_sanitizer=false
build_allreduce_only=false
collective_trace=true
install_dependencies=false
build_release=true
build_bfd=true
install_library=false
build_local_gpu_only=false
clean_build=true
npkit_enabled=false
build_package=false
build_freorg_bkwdcomp=true
run_tests=false
run_tests_all=false
build_static=false
build_tests=false
build_verbose=0
time_trace=false
enable_all_jobs=true
enable_ninja=""

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" --longoptions address-sanitizer,build_allreduce_only,dependencies,debug,disable_backtrace,disable-colltrace,fast,help,install,limit-nprocs,local_gpu_only,no_clean,npkit-enable,package_build,prefix:,rm-legacy-include-dir,run_tests_all,run_tests_quick,tests_build,time-trace,verbose --options hidptrs -- "$@")
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
         --address-sanitizer)        build_address_sanitizer=true;                                       shift ;;
         --build_allreduce_only)     build_allreduce_only=true;                                          shift ;;
    -d | --dependencies)             install_dependencies=true;                                          shift ;;
         --debug)                    build_release=false;                                                shift ;;
         --disable_backtrace)        build_bfd=false;                                                    shift ;;
         --disable-colltrace)        collective_trace=false;                                             shift ;;
         --fast)                     build_bfd=false; build_local_gpu_only=true; collective_trace=false; shift ;;
    -h | --help)                     display_help;                                                       exit 0 ;;
    -i | --install)                  install_library=true;                                               shift ;;
    -l | --limit-nprocs)             enable_all_jobs=false;                                              shift ;;
         --local_gpu_only)           build_local_gpu_only=true;                                          shift ;;
         --no_clean)                 clean_build=false;                                                  shift ;;
         --npkit-enable)             npkit_enabled=true;                                                 shift ;;
    -p | --package_build)            build_package=true;                                                 shift ;;
         --prefix)                   install_prefix=${2}                                                 shift 2 ;;
         --rm-legacy-include-dir)    build_freorg_bkwdcomp=false;                                        shift ;;
    -r | --run_tests_quick)          run_tests=true;                                                     shift ;;
         --run_tests_all)            run_tests=true; run_tests_all=true;                                 shift ;;
         --static)                   build_static=true;                                                  shift ;;
    -t | --tests_build)              build_tests=true;                                                   shift ;;
         --time-trace)               time_trace=true;                                                    shift ;;
         --verbose)                  build_verbose=1;                                                    shift ;;
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
    unit_test_path="./build/release/test/rccl-UnitTests"
else
    unit_test_path="./build/debug/test/rccl-UnitTests"
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

# Address sanitizer
if [[ "${build_address_sanitizer}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_ADDRESS_SANITIZER=ON"
fi

# AllReduce only
if [[ "${build_allreduce_only}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_ALLREDUCE_ONLY=ON"
fi

# Backtrace support
if [[ "${build_bfd}" == false ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_BFD=OFF"
fi

# Backward compatibility wrappers
if [[ "${build_freorg_bkwdcomp}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=ON"
else
    cmake_common_options="${cmake_common_options} -DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=OFF"
fi

# Build local GPU arch only
if [[ "$build_local_gpu_only" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_LOCAL_GPU_TARGET_ONLY=ON"
fi

# shared vs static
if [[ "${build_static}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_SHARED_LIBS=OFF"
fi

# Disable collective trace
if [[ "${collective_trace}" == false ]]; then
    cmake_common_options="${cmake_common_options} -DCOLLTRACE=OFF"
fi

# Install dependencies
if ($install_dependencies); then
    cmake_common_options="${cmake_common_options} -DINSTALL_DEPENDENCIES=ON"
fi

cmake_executable=cmake
case "${OS_ID}" in
    centos|rhel)
    cmake_executable=cmake3
  ;;
esac

npkit_options=""
if ($npkit_enabled); then
    npkit_options="-DENABLE_NPKIT \
    -DENABLE_NPKIT_EVENT_TIME_SYNC_GPU \
    -DENABLE_NPKIT_EVENT_TIME_SYNC_CPU \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_EXIT \
    -DENABLE_NPKIT_EVENT_COPY_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_COPY_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_DIRECT_COPY_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_DIRECT_COPY_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_DIRECT_RECV_ENTRY \
    -DENABLE_NPKIT_EVENT_DIRECT_RECV_EXIT \
    -DENABLE_NPKIT_EVENT_DIRECT_RECV_COPY_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_DIRECT_RECV_COPY_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_DIRECT_RECV_REDUCE_COPY_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_DIRECT_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_DIRECT_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_DIRECT_SEND_FROM_OUTPUT_ENTRY \
    -DENABLE_NPKIT_EVENT_DIRECT_SEND_FROM_OUTPUT_EXIT \
    -DENABLE_NPKIT_EVENT_RECV_ENTRY \
    -DENABLE_NPKIT_EVENT_RECV_EXIT \
    -DENABLE_NPKIT_EVENT_RECV_COPY_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_RECV_COPY_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_ENTRY \
    -DENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_EXIT \
    -DENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_RECV_REDUCE_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_RECV_REDUCE_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_SEND_FROM_OUTPUT_ENTRY \
    -DENABLE_NPKIT_EVENT_SEND_FROM_OUTPUT_EXIT \
    -DENABLE_NPKIT_EVENT_PRIM_SIMPLE_WAIT_PEER_ENTRY \
    -DENABLE_NPKIT_EVENT_PRIM_SIMPLE_WAIT_PEER_EXIT \
    -DENABLE_NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_ENTRY \
    -DENABLE_NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_EXIT \
    -DENABLE_NPKIT_EVENT_PRIM_LL_WAIT_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_PRIM_LL_WAIT_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY \
    -DENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT \
    -DENABLE_NPKIT_EVENT_PRIM_LL128_WAIT_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_PRIM_LL128_WAIT_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_ENTRY \
    -DENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_EXIT \
    -DENABLE_NPKIT_EVENT_NET_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_NET_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_NET_RECV_ENTRY \
    -DENABLE_NPKIT_EVENT_NET_RECV_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_EXIT \
    -DENABLE_NPKIT_EVENT_SEND_RECV_LOCAL_COPY_ENTRY \
    -DENABLE_NPKIT_EVENT_SEND_RECV_LOCAL_COPY_EXIT \
    -DENABLE_NPKIT_EVENT_SEND_RECV_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_SEND_RECV_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_SEND_RECV_RECV_ENTRY \
    -DENABLE_NPKIT_EVENT_SEND_RECV_RECV_EXIT \
    -DENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME"
fi

check_exit_code "$?"

if ($enable_all_jobs); then
    job_number=$(nproc)
else
    job_number=16
fi

if ($time_trace); then
    build_system="ninja"
    enable_ninja="-GNinja"
else
    build_system="make"
fi

if ($build_tests) || (($run_tests) && [[ ! -f ./test/rccl-UnitTests ]]); then
    CXX=$ROCM_BIN_PATH/hipcc $cmake_executable $cmake_common_options -DBUILD_TESTS=ON -DNPKIT_FLAGS="${npkit_options}" -DCMAKE_INSTALL_PREFIX=$ROCM_PATH -DROCM_PATH=$ROCM_PATH $enable_ninja ../../.
else
    CXX=$ROCM_BIN_PATH/hipcc $cmake_executable $cmake_common_options -DBUILD_TESTS=OFF -DNPKIT_FLAGS="${npkit_options}" -DCMAKE_INSTALL_PREFIX=$ROCM_PATH -DROCM_PATH=$ROCM_PATH $enable_ninja ../../.
fi
check_exit_code "$?"

if ($install_library); then
    VERBOSE=${build_verbose} $build_system -j $job_number install
else
    VERBOSE=${build_verbose} $build_system -j $job_number
fi
check_exit_code "$?"

if ($build_package); then
    make package
    check_exit_code "$?"
fi

# Optionally, run tests if they're enabled.
if ($run_tests); then
    if (test -f "./test/rccl-UnitTests"); then
        if ($run_tests_all); then
            ./test/rccl-UnitTests
        else
            ./test/rccl-UnitTests --gtest_filter="AllReduce.*"
        fi
    else
        echo "rccl unit tests have not been built yet; please re-run script with -t to build rccl unit tests."
        exit 1
    fi
fi

if ($time_trace); then
    search_dir="../../"
    time_trace_dir=$(find "$search_dir" -type d -name "time-trace" -print -quit)

    if [ "$time_trace_dir" ]; then
        time_trace_script="$time_trace_dir/rccl-TimeTrace.sh"
        if [ -x "$time_trace_script" ]; then
            echo "Generating RCCL-compile-timeline.html..."
            (cd "$time_trace_dir" && ./rccl-TimeTrace.sh)
        else
            echo "Error: Unable to execute $time_trace_script. Make sure the file has the correct permissions."
        fi
    else
        echo "Error: time-trace folder not found in $search_dir."
    fi
fi