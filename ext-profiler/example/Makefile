#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
.DEFAULT_GOAL: build
include ../../makefiles/common.mk
BUILDDIR ?= .
NCCLDIR  := $(BUILDDIR)

SRC_FILES := $(wildcard *.cc)
DST_DIR   := $(BUILDDIR)
OBJ_FILES := $(SRC_FILES:%.cc=${DST_DIR}/%.o)
DEP_FILES := $(OBJ_FILES:%.o=%.dep)

build: ${BUILDDIR}/librccl-profiler.so

${BUILDDIR}/librccl-profiler.so: ${SRC_FILES}
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) -Inccl -I${CUDA_INC} -fPIC -shared -o $@ $^

clean:
	rm -f ${BUILDDIR}/librccl-profiler.so
