#!/bin/bash
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.

set -eu

# Make this directory the PWD
cd "$(dirname "${BASH_SOURCE[0]}")"

# Build doxygen info
./run_doxygen.sh

# Build sphinx docs
cd source
make clean
make html
