# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

# Set to where RCCL is installed
RCCL_INSTALL=${PWD}/../../build


HIP_PATH?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
HIP_PATH=../../..
endif

HIPCC=/opt/rocm/bin/hipcc

EXE=MemleakTest
# adding debug mode 
CXXFLAGS = -std=c++11 -g -O3 -I../../src/include -I$(RCCL_INSTALL) -L$(RCCL_INSTALL) -lrccl


all: $(EXE)

$(EXE): $(EXE).cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o $(EXE)