RCCL_DIR ?= ../../build/release
MPI_INC_DIR ?= /usr/include/x86_64-linux-gnu/mpi
MPI_LIB_DIR ?= /usr/lib/x86_64-linux-gnu

INCLUDES = -I$(MPI_INC_DIR) -I$(MPI_DIR)/include -I$(RCCL_DIR) -I$(RCCL_DIR)/include/rccl
LDFLAGS  = -L$(MPI_LIB_DIR) -L$(MPI_DIR)/lib -L$(RCCL_DIR) -lmpi -lrccl

ifdef MPI_DIR
main: rcclReplayer.cpp
	/opt/rocm/bin/hipcc rcclReplayer.cpp -O1 -g -o rcclReplayer $(INCLUDES) $(LDFLAGS)
else
main:
	@echo "Error: MPI_DIR was not specified."
	@exit 1
endif

clean:
	rm -f ./rcclReplayer
