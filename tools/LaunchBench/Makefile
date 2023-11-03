ROCM_DIR = /opt/rocm
HIPCC    = $(ROCM_DIR)/bin/hipcc
CCFLAGS  = -lhsa-runtime64 -fopenmp

all: LaunchBench

LaunchBench: LaunchBench.cpp Common.hpp
	$(HIPCC) $(CCFLAGS) $< -o $@

clean:
	rm -f ./LaunchBench
