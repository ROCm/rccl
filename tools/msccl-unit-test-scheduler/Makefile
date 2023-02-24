# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

ROCM_HOME ?= "/opt/rocm"
RCCL_BIN_HOME ?= ""
RCCL_SRC_HOME ?= ""
INC := -I$(ROCM_HOME)/include -I$(RCCL_BIN_HOME)/include -I$(RCCL_SRC_HOME)/src/include
LNK := -L$(RCCL_BIN_HOME) -lrccl
SCHEDULER_SO := libmsccl-scheduler.so

default: $(SCHEDULER_SO)

$(SCHEDULER_SO): scheduler.cc parser.cc
	$(CXX) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(SCHEDULER_SO) $^ $(LNK)

clean:
	rm -f $(SCHEDULER_SO)
