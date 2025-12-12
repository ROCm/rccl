# Makefile for RcclReplayer tests

HIPCC = hipcc

ROCM_DIR ?= /opt/rocm
RCCL_DIR ?= ../../../build/release
MPI_DIR  ?= /opt/ompi

CXXFLAGS = -std=c++20 -O2 -g
INCLUDES = -I../../../src/include -I$(MPI_DIR)/include -I$(ROCM_DIR)/include
LDFLAGS = -L$(RCCL_DIR) -L$(MPI_DIR)/lib -L$(ROCM_DIR)/lib
LIBS = -lrccl -lmpi -lamdhip64 -lpthread
RPATH = -Wl,-rpath,$(RCCL_DIR) -Wl,-rpath,$(MPI_DIR)/lib -Wl,-rpath,$(ROCM_DIR)/lib

TARGET = test_multi_thread_single_gpu

all: $(TARGET)

$(TARGET): test_multi_thread_single_gpu.cpp
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS) $(RPATH)

clean:
	rm -f $(TARGET)

.PHONY: all clean
