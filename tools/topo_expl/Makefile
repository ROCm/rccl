# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
HIP_PATH ?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
HIP_PATH = ../../..
endif
HIPCC = $(HIP_PATH)/bin/hipcc

EXE = topo_expl
CXXFLAGS = -g -O3 -Iinclude -I../../src -I../../src/include -I../../src/graph/ -I/opt/rocm/include/ -DTOPO_EXPL -DENABLE_TRACE

files = $(EXE).cpp model.cpp utils.cpp ../../src/graph/topo.cc ../../src/graph/rings.cc ../../src/graph/paths.cc ../../src/graph/trees.cc ../../src/misc/param.cc \
	../../src/graph/search.cc ../../src/graph/connect.cc ../../src/graph/tuning.cc ../../src/graph/xml.cc ../../src/misc/nvmlwrap_stub.cc ../../src/graph/rome_models.cc

all: $(EXE)

$(EXE): $(files)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

clean:
	rm -f *.o $(EXE)
