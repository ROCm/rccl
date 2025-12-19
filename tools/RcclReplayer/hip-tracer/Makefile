ROCM_DIR ?= /opt/rocm
RCCL_DIR ?= ../../../build/release

INCLUDES = -I$(ROCM_DIR)/include -I.
LDFLAGS  = -L$(ROCM_DIR)/lib -L$(RCCL_DIR) -lroctracer64 -lrccl -Wl,-rpath,'$$ORIGIN/../../../build/release'

main: plugin.cpp rccl_hip_api.h
	$(ROCM_DIR)/bin/hipcc -fPIC -shared -O2 plugin.cpp -o librccl-hip-tracer.so $(INCLUDES) $(LDFLAGS)

clean:
	rm -f ./librccl-hip-tracer.so
