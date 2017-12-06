RCCL_INSTALL_DIR=/opt/rocm/rccl
HIP_DIR=/opt/rocm/hip
all: lib install


lib:
	/opt/rocm/bin/hcc -I../inc -I${HIP_DIR}/include `/opt/rocm/bin/hcc-config --shared --cxxflags --ldflags` -L/opt/rocm/hip/lib -lhip_device rccl.cpp -o librccl.so

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
