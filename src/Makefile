RCCL_INSTALL_DIR=/opt/rocm/rccl
HIP_DIR=/opt/rocm/hip
TARGETS=--amdgpu-target=gfx803 --amdgpu-target=gfx900
all: lib install


lib:
	/opt/rocm/bin/hcc -I../inc ${TARGETS} -I${HIP_DIR}/include `/opt/rocm/bin/hcc-config --shared --cxxflags --ldflags` -L/opt/rocm/hip/lib -lhip_device rccl.cpp -o librccl.so


lib-debug:
	/opt/rocm/bin/hcc -I../inc ${TARGETS} -DRCCL_DEBUG=1 -I${HIP_DIR}/include `/opt/rocm/bin/hcc-config --shared --cxxflags --ldflags` -L/opt/rocm/hip/lib -lhip_device rccl.cpp -o librccl.so

install:
	mkdir -p $(RCCL_INSTALL_DIR)
	mkdir -p $(RCCL_INSTALL_DIR)/include
	mkdir -p $(RCCL_INSTALL_DIR)/lib
	cp librccl.so $(RCCL_INSTALL_DIR)/lib
	cp ../inc/rccl.h $(RCCL_INSTALL_DIR)/include
	cp rcclCheck.h $(RCCL_INSTALL_DIR)/include

clean:
	rm -rf librccl.so

uninstall:
	rm -rf $(RCCL_INSTALL_DIR)
