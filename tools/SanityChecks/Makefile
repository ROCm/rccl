# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

# Set to where RCCL is installed
RCCL_INSTALL=/home/apotnuru/bugs/rccl/build/release


HIP_PATH?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=MemleakTest
# adding debug mode 
CXXFLAGS = -std=c++14 -O3 -I../../src/include -I$(RCCL_INSTALL) -L$(RCCL_INSTALL) -lrccl


all: $(EXE)

$(EXE): $(EXE).cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o $(EXE)