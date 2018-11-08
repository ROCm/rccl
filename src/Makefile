SHELL := /bin/bash
RCCL_INSTALL_DIR=/opt/rocm/rccl
HIP_DIR=/opt/rocm/hip
HCC_DIR=/opt/rocm/hcc
HSA_DIR=/opt/rocm/hsa
TARGETS=--amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906
SRC=rccl.cpp rcclAllReduce.cpp rcclBcast.cpp rcclReduce.cpp rcclTracker.cpp rcclAllGather.cpp

all: lib

../inc/rccl/rccl-version.h: ../inc/rccl/rccl-version.h.in
	gitver=`git log -1 --pretty='%h'`; \
	string=`cat ../CMakeLists.txt`; \
	re='    VERSION.([0-9]*)\.([0-9]*)\.([0-9]*).'; \
	if [[ $${string} =~ $${re} ]]; then \
	sed -e "s/@rccl_VERSION_MAJOR@/$${BASH_REMATCH[1]}/g" \
	    -e "s/@rccl_VERSION_MINOR@/$${BASH_REMATCH[2]}/g" \
	    -e "s/@rccl_VERSION_PATCH@/$${BASH_REMATCH[3]}/g" \
	    -e "s/@rccl_VERSION_TWEAK@/$${gitver}/g" \
            $< > $@; \
	fi

lib: ../inc/rccl/rccl-version.h
	${HCC_DIR}/bin/hcc -I../inc ${TARGETS} -I${HIP_DIR}/include -I${HSA_DIR}/include/ `${HCC_DIR}/bin/hcc-config --shared --cxxflags --ldflags` -L${HIP_DIR}/lib -lhip_hcc $(SRC) -o librccl.so

install:
	mkdir -p $(RCCL_INSTALL_DIR)
	mkdir -p $(RCCL_INSTALL_DIR)/include
	mkdir -p $(RCCL_INSTALL_DIR)/include/rccl
	mkdir -p $(RCCL_INSTALL_DIR)/lib
	cp librccl.so $(RCCL_INSTALL_DIR)/lib
	cp ../inc/rccl/rccl.h $(RCCL_INSTALL_DIR)/include/rccl
	cp ../inc/rccl/rccl-version.h $(RCCL_INSTALL_DIR)/include/rccl

clean:
	rm -rf librccl.so
	rm ../inc/rccl/rccl-version.h

uninstall:
	rm -rf $(RCCL_INSTALL_DIR)
