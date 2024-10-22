cd ../../
rm -rf build
mkdir build
cd build

# Specify the correct path to the CMake binary
CMAKE_BIN_PATH=../cmake-3.28.2/bin/cmake

if ! $CMAKE_BIN_PATH .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm; then
    echo "CMake configuration failed!"
    exit 1
fi

if ! make -j$(nproc); then
    echo "Build failed!"
    exit 1
fi

echo "Build completed"
