LIB_PATH_DIR=/home/apotnuru/bugs/rccl/build/release
echo $LIB_PATH_DIR
LD_LIBRARY_PATH=$LIB_PATH_DIR valgrind --leak-check=full ./MemleakTest 0 1