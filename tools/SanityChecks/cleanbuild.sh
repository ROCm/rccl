#!/bin/bash

# Navigate to the project root
cd ../../ || exit 1

# Clean up any previous builds and create a fresh build directory
rm -rf build
mkdir build
cd build || exit 1

# Specify the path to the CMake binary
CMAKE_BIN_PATH="../cmake-3.28.2/bin/cmake"

# Run CMake with the desired configuration
if ! "$CMAKE_BIN_PATH" -DCMAKE_PREFIX_PATH=/home/apotnuru/bugs/rccl -DCMAKE_BUILD_TYPE=Debug ..; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build the project with all available CPU cores
if ! make -j"$(nproc)"; then
    echo "Build failed!"
    exit 1
fi

echo "Build completed successfully!"
