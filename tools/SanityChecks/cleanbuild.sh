# #!/bin/bash

# # Navigate to the project root
# cd ../../ || exit 1

# # Clean up any previous builds and create a fresh build directory
# rm -rf build || exit 1

#uses install.sh script for building rccl library
#./install.sh --debug --disable-mscclpp --local_gpu_only || exit 1

#echo "Build completed successfully!"
cd ../../
rm -rf build
mkdir build
cd build

# Specify the correct path to the CMake binary
CMAKE_BIN_PATH=cmake #../cmake-3.28.2/bin/cmake
# -DCMAKE_BUILD_TYPE=Debug

if ! $CMAKE_BIN_PATH ..  -DCMAKE_PREFIX_PATH=/opt/rocm; then
    echo "CMake configuration failed!"
    exit 1
fi

if ! make -j$(nproc); then
    echo "Build failed!"
    exit 1
fi

echo "Build completed"