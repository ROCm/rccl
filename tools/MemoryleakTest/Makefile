# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

# Set to where RCCL is installed
RCCL_INSTALL=../../build

HIP_PATH?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=MemleakTest
# adding debug mode 
CXXFLAGS = -std=c++14 -g -O3 -I../../src/include -I$(RCCL_INSTALL) -L$(RCCL_INSTALL) -lrccl
#CXXFLAGS = -std=c++14 -g -O3 -I../../src/include -L../../build -lrccl


all: $(EXE)

$(EXE): $(EXE).cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o $(EXE)

#/opt/rocm/bin/hipcc -std=c++14 -g -O3 -I/home/apotnuru/bugs/rccl/include -L/home/apotnuru/bugs/rccl/build -lrccl MemleakTest.cpp -o MemleakTest

