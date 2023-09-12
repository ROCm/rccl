# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
HIP_PATH ?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
HIP_PATH = ../../..
endif
HIPCC = $(HIP_PATH)/bin/hipcc

EXE = topo_expl
CXXFLAGS = -g -Iinclude -Ihipify_rccl/include -Ihipify_rccl/graph -I/opt/rocm/include/ -DTOPO_EXPL -DENABLE_TRACE -DNVTX_NO_IMPL

files = $(EXE).cpp model.cpp utils.cpp hipify_rccl/graph/topo.cc hipify_rccl/graph/rings.cc hipify_rccl/graph/paths.cc hipify_rccl/graph/trees.cc ../../src/misc/param.cc \
	hipify_rccl/graph/search.cc hipify_rccl/graph/connect.cc hipify_rccl/graph/tuning.cc hipify_rccl/graph/xml.cc ../../src/misc/nvmlwrap_stub.cc hipify_rccl/graph/rome_models.cc hipify_rccl/graph/archinfo.cc

all: hipify $(EXE)

$(EXE): $(files)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

hipify:
	rm -rf hipify_rccl
	mkdir -p hipify_rccl
	cp -a ../../src/include/ hipify_rccl/
	cp -a ../../src/graph/ hipify_rccl/
	cp -ar ../../src/misc/archinfo.cc hipify_rccl/graph/
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/*.h
	hipify-perl -inplace -quiet-warnings hipify_rccl/graph/*

clean:
	rm -rf hipify_rccl
	rm -f *.o $(EXE)
