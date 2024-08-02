#!/bin/bash
# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.

# #################################################
# global variables
# #################################################
ROCM_PATH=${ROCM_PATH:="/opt/rocm"}

# Default values
build_address_sanitizer=false
build_bfd=false
build_freorg_bkwdcomp=false
build_local_gpu_only=false
build_amdgpu_targets=""
build_package=false
build_release=true
build_static=false
build_tests=false
build_verbose=false
clean_build=true
collective_trace=true
enable_ninja=""
install_dependencies=false
install_library=false
install_prefix="${ROCM_PATH}"
msccl_kernel_enabled=true
mscclpp_enabled=false
num_parallel_jobs=$(nproc)
npkit_enabled=false
openmp_test_enabled=false
roctx_enabled=false
run_tests=false
run_tests_all=false
time_trace=false

# #################################################
# helper functions
# #################################################
function display_help()
{
    echo "RCCL build & installation helper script"
    echo " Options:"
    echo "       --address-sanitizer     Build with address sanitizer enabled"
    echo "    -d|--dependencies          Install RCCL depdencencies"
    echo "       --debug                 Build debug library"
    echo "       --enable_backtrace      Build with custom backtrace support"
    echo "       --disable-colltrace     Build without collective trace"
    echo "       --disable-msccl-kernel  Build without MSCCL kernels"
    echo "       --disable-mscclpp       Build without MSCCL++ support"
    echo "    -f|--fast                  Quick-build RCCL (local gpu arch only, no backtrace, and collective trace support)"
    echo "    -h|--help                  Prints this help message"
    echo "    -i|--install               Install RCCL library (see --prefix argument below)"
    echo "    -j|--jobs                  Specify how many parallel compilation jobs to run ($num_parallel_jobs by default)"
    echo "    -l|--local_gpu_only        Only compile for local GPU architecture"
    echo "       --amdgpu_targets        Only compile for specified GPU architecture(s). For multiple targets, seperate by ';' (builds for all supported GPU architectures by default)"
    echo "       --no_clean              Don't delete files if they already exist"
    echo "       --npkit-enable          Compile with npkit enabled"
    echo "       --openmp-test-enable    Enable OpenMP in rccl unit tests"
    echo "       --roctx-enable          Compile with roctx enabled (example usage: rocprof --roctx-trace ./rccl-program)"
    echo "    -p|--package_build         Build RCCL package"
    echo "       --prefix                Specify custom directory to install RCCL to (default: \`/opt/rocm\`)"
    echo "       --rm-legacy-include-dir Remove legacy include dir Packaging added for file/folder reorg backward compatibility"
    echo "       --run_tests_all         Run all rccl unit tests (must be built already)"
    echo "    -r|--run_tests_quick       Run small subset of rccl unit tests (must be built already)"
    echo "       --static                Build RCCL as a static library instead of shared library"
    echo "    -t|--tests_build           Build rccl unit tests, but do not run"
    echo "       --time-trace            Plot the build time of RCCL (requires \`ninja-build\` package installed on the system)"
    echo "       --verbose               Show compile commands"
}

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ "$?" -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" --options dfhij:lprt --longoptions address-sanitizer,dependencies,debug,enable_backtrace,disable-colltrace,disable-msccl-kernel,disable-mscclpp,fast,help,install,jobs:,local_gpu_only,amdgpu_targets:,no_clean,npkit-enable,openmp-test-enable,roctx-enable,package_build,prefix:,rm-legacy-include-dir,run_tests_all,run_tests_quick,static,tests_build,time-trace,verbose -- "$@")
else
    echo "Need a new version of getopt"
    exit 1
fi

if [[ "$?" -ne 0 ]]; then
    echo "getopt invocation failed; could not parse the command line";
    exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
    case "${1}" in
         --address-sanitizer)        build_address_sanitizer=true;                                                                     shift ;;
    -d | --dependencies)             install_dependencies=true;                                                                        shift ;;
         --debug)                    build_release=false;                                                                              shift ;;
         --enable_backtrace)         build_bfd=true;                                                                                   shift ;;
         --disable-colltrace)        collective_trace=false;                                                                           shift ;;
         --disable-msccl-kernel)     msccl_kernel_enabled=false;                                                                       shift ;;
         --disable-mscclpp)          mscclpp_enabled=false;                                                                            shift ;;
    -f | --fast)                     build_local_gpu_only=true; collective_trace=false; msccl_kernel_enabled=false;                    shift ;;
    -h | --help)                     display_help;                                                                                     exit 0 ;;
    -i | --install)                  install_library=true;                                                                             shift ;;
    -j | --jobs)                     num_parallel_jobs=${2};                                                                           shift 2 ;;
    -l | --local_gpu_only)           build_local_gpu_only=true;                                                                        shift ;;
         --amdgpu_targets)           build_amdgpu_targets=${2};                                                                        shift 2 ;;
         --no_clean)                 clean_build=false;                                                                                shift ;;
         --npkit-enable)             npkit_enabled=true;                                                                               shift ;;
         --openmp-test-enable)       openmp_test_enabled=true;                                                                         shift ;;
         --roctx-enable)             roctx_enabled=true;                                                                               shift ;;
    -p | --package_build)            build_package=true;                                                                               shift ;;
         --prefix)                   install_library=true; install_prefix=${2};                                                        shift 2 ;;
         --rm-legacy-include-dir)    build_freorg_bkwdcomp=false;                                                                      shift ;;
    -r | --run_tests_quick)          run_tests=true;                                                                                   shift ;;
         --run_tests_all)            run_tests=true; run_tests_all=true;                                                               shift ;;
         --static)                   build_static=true;                                                                                shift ;;
    -t | --tests_build)              build_tests=true;                                                                                 shift ;;
         --time-trace)               time_trace=true;                                                                                  shift ;;
         --verbose)                  build_verbose=true;                                                                               shift ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
    esac
done

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

# CMake executable
cmake_executable=cmake
time_trace_ninja_msg="apt-get install ninja-build"
case "${OS_ID}" in
    centos|rhel)
    cmake_executable=cmake3
    time_trace_ninja_msg="dnf install ninja-build"
  ;;
esac

# CMake build options; starts with toolchain info
cmake_common_options="--toolchain=toolchain-linux.cmake"

# throw error code after running a command in the install script
check_exit_code( )
{
    if (( $1 != 0 )); then
        exit "$1"
    fi
}

# set RCCL-UnitTests path
if [[ "${build_release}" == true ]]; then
    unit_test_path="./build/release/test/rccl-UnitTests"
else
    unit_test_path="./build/debug/test/rccl-UnitTests"
fi

if [[ "${run_tests}" == true ]] && [[ -f "${unit_test_path}" ]]; then
    if [[ "${build_tests}" == false ]]; then
        clean_build=false
    fi
fi

# #################################################
# prep
# #################################################
# ensure a clean build environment
if [[ "${clean_build}" == true ]]; then
    if [[ "${build_release}" == true ]]; then
        rm -rf build/release
    else
        rm -rf build/debug
    fi
fi

# Create and go to the build directory.
mkdir -p build; cd build

# Create and go to build type directory
if [[ "${build_release}" == true ]]; then
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

# Backtrace support
if [[ "${build_bfd}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_BFD=ON"
fi

# Backward compatibility wrappers
if [[ "${build_freorg_bkwdcomp}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=ON"
fi

# Build local GPU arch only
if [[ "${build_local_gpu_only}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_LOCAL_GPU_TARGET_ONLY=ON"
fi

# Build for specified GPU target(s) only
if [[ ! -z "${build_amdgpu_targets}" ]]; then
    cmake_common_options="${cmake_common_options} -DAMDGPU_TARGETS=${build_amdgpu_targets}"
fi

# shared vs static
if [[ "${build_static}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_SHARED_LIBS=OFF"
fi

# Disable collective trace
if [[ "${collective_trace}" == false ]]; then
    cmake_common_options="${cmake_common_options} -DCOLLTRACE=OFF"
fi

# Disable msccl kernel
if [[ "${msccl_kernel_enabled}" == false ]]; then
    cmake_common_options="${cmake_common_options} -DENABLE_MSCCL_KERNEL=OFF"
fi

if [[ "${mscclpp_enabled}" == false ]]; then
    cmake_common_options="${cmake_common_options} -DENABLE_MSCCLPP=OFF"
fi

# Install dependencies
if [[ "${install_dependencies}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DINSTALL_DEPENDENCIES=ON"
fi

# Install RCCL library
if [[ "${install_library}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DCMAKE_INSTALL_PREFIX=${install_prefix}"
fi

# Enable ROCTX
if [[ "${roctx_enabled}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DROCTX=ON"
fi

# Enable OpenMP in unit tests
if [[ "${openmp_test_enabled}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DOPENMP_TESTS_ENABLED=ON"
fi

# Enable NPKit
npkit_options=""
if [[ "${npkit_enabled}" == true ]]; then
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
    -DENABLE_NPKIT_EVENT_NET_TEST_ENTRY \
    -DENABLE_NPKIT_EVENT_NET_TEST_EXIT \
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
    -DENABLE_NPKIT_EVENT_ALL_GATHER_RING_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_GATHER_RING_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY \
    -DENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT \
    -DENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_ENTRY \
    -DENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_EXIT \
    -DENABLE_NPKIT_EVENT_MSCCL_REDUCE_ENTRY \
    -DENABLE_NPKIT_EVENT_MSCCL_REDUCE_EXIT \
    -DENABLE_NPKIT_EVENT_MSCCL_SEND_ENTRY \
    -DENABLE_NPKIT_EVENT_MSCCL_SEND_EXIT \
    -DENABLE_NPKIT_EVENT_MSCCL_RECV_ENTRY \
    -DENABLE_NPKIT_EVENT_MSCCL_RECV_EXIT \
    -DENABLE_NPKIT_EVENT_MSCCL_RUN_ENTRY \
    -DENABLE_NPKIT_EVENT_MSCCL_RUN_EXIT \
    -DENABLE_NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_ENTRY \
    -DENABLE_NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_EXIT \
    -DENABLE_NPKIT_EVENT_MSCCL_INIT_ENTRY \
    -DENABLE_NPKIT_EVENT_MSCCL_INIT_EXIT \
    -DENABLE_NPKIT_EVENT_BROADCAST_RING_ENTRY \
    -DENABLE_NPKIT_EVENT_BROADCAST_RING_EXIT \
    -DENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME"
fi

check_exit_code "$?"

# Enable ninja build for time tracing
if [[ "${time_trace}" == true ]]; then
    if ! hash ninja &>/dev/null ; then
        echo "ninja could not be found"
        echo "Use \"${time_trace_ninja_msg}\" to install ninja"
        exit 1
    fi
    build_system="ninja"
    enable_ninja="-GNinja"
else
    build_system="make"
fi

# Add common CMake options
cmake_common_options="${cmake_common_options} -DROCM_PATH=${ROCM_PATH} -DONLY_FUNCS=${ONLY_FUNCS} ${enable_ninja}"

# Build RCCL-UnitTests, if enabled
if [[ "${build_tests}" == true ]] || ([[ "${run_tests}" == true ]] && [[ ! -x ./test/rccl-UnitTests ]]); then
    cmake_common_options="${cmake_common_options} -DBUILD_TESTS=ON"
fi

# Initiate RCCL CMake
# Passing NPKIT_FLAGS separately (not as part of ${cmake_common_options}) as
# ${npkit_options} need to be passed "as-is" i.e. with `-D` to CMakeLists.txt
${cmake_executable} ${cmake_common_options} -DNPKIT_FLAGS="${npkit_options}" ../../.
check_exit_code "$?"

# Enable verbose output from Makefile
if [[ "${build_verbose}" == true ]]; then
    build_system="${build_system} VERBOSE=1"
fi

# Initiate RCCL build (and install)
if [[ "${install_library}" == true ]]; then
    ${build_system} -j ${num_parallel_jobs} install
else
    ${build_system} -j ${num_parallel_jobs}
fi
check_exit_code "$?"

# Initiate package build with `make package`, if enabled
if [[ "${build_package}" == true ]]; then
    make package
    check_exit_code "$?"
fi

# Optionally, run RCCL-UnitTests, if they're enabled.
if [[ "${run_tests}" == true ]]; then
    if [[ -x "./test/rccl-UnitTests" ]]; then
        if [[ "${run_tests_all}" == true ]]; then
            ./test/rccl-UnitTests
        else
            ./test/rccl-UnitTests --gtest_filter="AllReduce.*"
        fi
    else
        echo "RCCL-UnitTests have not been built yet; Please re-run script with \"-t\" to build RCCL-UnitTests."
        exit 1
    fi
fi

# Generate time trace for RCCL build using tools/time-trace
if [[ "${time_trace}" == true ]]; then
    search_dir="../../tools"
    time_trace_dir=$(find "${search_dir}" -type d -name "time-trace" -print -quit)

    if [[ -n "${time_trace_dir}" ]]; then
        time_trace_script="${time_trace_dir}/rccl-TimeTrace.sh"
        if [[ -x "${time_trace_script}" ]]; then
            echo "Generating RCCL-compile-timeline.html..."
            (cd "${time_trace_dir}" && ./rccl-TimeTrace.sh)
        else
            echo "Error: Unable to execute ${time_trace_script}. Make sure the file has the correct permissions."
        fi
    else
        echo "Error: time-trace folder not found in ${search_dir}."
    fi
fi
