make clean
make || exit 1

echo -e "\n\n End of compiling the test program, starting memory leak check \n\n"

LIB_PATH_DIR=$PWD/../../build/debug
echo $LIB_PATH_DIR
LD_LIBRARY_PATH=$LIB_PATH_DIR NCCL_DEBUG=INFO valgrind --leak-check=full ./MemleakTest 0 1