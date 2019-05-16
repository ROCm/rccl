#!/bin/bash
# Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

export RCCL_DIR=$PWD/rccl-internal
export RCCL_INSTALL=$PWD/rccl-install
export ROCM_PATH=/opt/rocm/bin

# #################################################
# helper functions
# #################################################
function display_help()
{
    echo "RCCL build & installation helper script"
    echo "./install [-h|--help] "
    echo "    [-h|--help] prints this help message"
    echo "    [-t|--test] run RCCL unit tests too"
}

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,clients,debug,test --options hicdt -- "$@")
else
    echo "Need a new version of getopt"
    exit 1
fi

if [[ $? -ne 0 ]]; then
    echo "getopt invocation failed; could not parse the command line";
    exit 1
fi

eval set -- "${GETOPT_PARSE}"

run_tests=false

while true; do
    case "${1}" in
	-h|--help)
	    display_help
	    exit 0
	    ;;
	-t|--test)
	    run_tests=true
	    shift ;;
	--) shift ; break ;;
	*)  echo "Unexpected command line parameter received; aborting";
	    exit 1
	    ;;
    esac
    done

# Install the pre-commit hook
#bash ./githooks/install

rm -rf build
mkdir build
cd build
CXX=$ROCM_PATH/hcc cmake -DCMAKE_INSTALL_PREFIX=$RCCL_INSTALL ..
make -j 8 install

if ($run_tests); then
# Optionally, run tests if they're enabled.
HSA_FORCE_FINE_GRAIN_PCIE=1 $RCCL_INSTALL/test/UnitTests
fi
