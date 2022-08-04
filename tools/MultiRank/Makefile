# Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.

# Set to where RCCL is installed
RCCL_INSTALL=../../build/release

HIP_PATH?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=rccl-allreduce-multirank rccl-reducescatter-multirank
CXXFLAGS = -std=c++11 -O3 -I$(RCCL_INSTALL)/include/rccl/ -L$(RCCL_INSTALL) -lrccl

all: $(EXE)

rccl-allreduce-multirank: rccl-allreduce-multirank.cc $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o $@

rccl-reducescatter-multirank: rccl-reducescatter-multirank.cc $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o $(EXE)
