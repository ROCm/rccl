RCCL_INSTALL_DIR=/opt/rocm/rccl
HIP_DIR=/opt/rocm/hip
HCC_DIR=/opt/rocm/hcc
TARGETS=--amdgpu-target=gfx803 --amdgpu-target=gfx900
SRC=rccl.cpp rcclAllReduce.cpp rcclBcast.cpp rcclReduce.cpp rcclTracker.cpp

all: lib

lib:
	${HCC_DIR}/bin/hcc -I../inc ${TARGETS} -I${HIP_DIR}/include `${HCC_DIR}/bin/hcc-config --shared --cxxflags --ldflags` -L${HIP_DIR}/lib -lhip_device -lhip_hcc $(SRC) -o librccl.so

install:
	mkdir -p $(RCCL_INSTALL_DIR)
	mkdir -p $(RCCL_INSTALL_DIR)/include
	mkdir -p $(RCCL_INSTALL_DIR)/include/rccl
	mkdir -p $(RCCL_INSTALL_DIR)/lib
	cp librccl.so $(RCCL_INSTALL_DIR)/lib
	cp ../inc/rccl/rccl.h $(RCCL_INSTALL_DIR)/include/rccl

clean:
	rm -rf librccl.so

uninstall:
	rm -rf $(RCCL_INSTALL_DIR)
