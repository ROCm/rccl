RCCL_DIR ?= ../../build/release

ifdef MPI_DIR
main: rcclReplayer.cpp
	/opt/rocm/bin/hipcc rcclReplayer.cpp \
		-g \
		-o rcclReplayer \
		-I$(MPI_DIR)/ \
		-I$(RCCL_DIR) \
		-I$(RCCL_DIR)/include/rccl \
		-I/opt/rocm/include/hip \
		-L$(MPI_DIR)/lib \
		-L$(RCCL_DIR) -lmpich -lrccl
else
main:
	@echo "Error: MPI_DIR was not specified."
	@exit 1
endif

clean:
	rm -f ./rcclReplayer