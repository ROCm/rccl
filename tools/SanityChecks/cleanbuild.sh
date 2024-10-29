#!/bin/bash

# Navigate to the project root
cd ../../ || exit 1

# Clean up any previous builds and create a fresh build directory
rm -rf build || exit 1

#uses install.sh script for building rccl library
./install.sh --debug --disable-mscclpp --local_gpu_only || exit 1
echo "Build completed successfully!"