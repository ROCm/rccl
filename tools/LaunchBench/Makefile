ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda
HIPCC     = $(ROCM_PATH)/bin/hipcc
NVCC      = $(CUDA_PATH)/bin/nvcc

CCFLAGS   = -lhsa-runtime64 -fopenmp -lnuma -g
NVFLAGS   = -O3  -x cu -lnuma -Xcompiler -fopenmp -gencode=arch=compute_90,code=sm_90

#ifeq ("$(shell test -e $(NVCC) && echo found)", "found")



all: LaunchBench

LaunchBench: LaunchBench.cpp Common.hpp Timeline.hpp
ifeq ("$(shell test -e $(NVCC) && echo found)", "found")
	$(NVCC) $(NVFLAGS) $< -o $@
else
	$(HIPCC) $(CCFLAGS) $< -o $@
endif

clean:
	rm -f ./LaunchBench
