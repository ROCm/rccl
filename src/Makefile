SHELL := /bin/bash
RCCL_INSTALL_DIR=/opt/rocm/rccl
HIP_DIR=/opt/rocm/hip
HCC_DIR=/opt/rocm/hcc
HSA_DIR=/opt/rocm/hsa
TARGETS=--amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906
SRC=rccl.cpp rcclAllReduce.cpp rcclBcast.cpp rcclReduce.cpp rcclTracker.cpp rcclAllGather.cpp

all: lib

lib:
	${HCC_DIR}/bin/hcc -I../inc ${TARGETS} -I${HIP_DIR}/include -I${HSA_DIR}/include/ `${HCC_DIR}/bin/hcc-config --shared --cxxflags --ldflags` -L${HIP_DIR}/lib -lhip_hcc $(SRC) -o librccl.so

install:
	mkdir -p $(RCCL_INSTALL_DIR)
	mkdir -p $(RCCL_INSTALL_DIR)/include
	mkdir -p $(RCCL_INSTALL_DIR)/lib
	cp librccl.so $(RCCL_INSTALL_DIR)/lib
	cp ../inc/rccl.h $(RCCL_INSTALL_DIR)/include
	cp ../inc/rccl-version.h $(RCCL_INSTALL_DIR)/include

clean:
	rm -rf librccl.so
	rm ../inc/rccl-version.h

uninstall:
	rm -rf $(RCCL_INSTALL_DIR)
