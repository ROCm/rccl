#!/bin/bash
# # Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.

set -eu

# Make this directory the PWD
cd "$(dirname "${BASH_SOURCE[0]}")"

# Rename our input file
cp ../src/nccl.h.in nccl.h

# Build the doxygen info
rm -rf docBin
doxygen Doxyfile

# Cleanup
rm nccl.h
