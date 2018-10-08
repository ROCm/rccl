all: comm bcast allreduce reduce multistream

ROCM_PATH=/opt/rocm
TEST_INC=../
RCCL_INC=$(ROCM_PATH)/rccl/include
RCCL_LIB=$(ROCM_PATH)/rccl/lib
HIPCC=$(ROCM_PATH)/hip/bin/hipcc
ARCHS=--amdgpu-target=gfx900 --amdgpu-target=gfx803 --amdgpu-target=gfx906

bcast: rcclBcast.cpp
	mkdir -p bin
	$(HIPCC) -I$(RCCL_INC) -I$(TEST_INC) $(ARCHS) rcclBcast.cpp -L$(RCCL_LIB) -lrccl -o ./bin/bcast

comm: rcclCommBcast.cpp
	mkdir -p bin
	$(HIPCC) -I$(RCCL_INC) -I$(TEST_INC) $(ARCHS) rcclCommBcast.cpp -L$(RCCL_LIB) -lrccl -o ./bin/commBcast


allreduce: rcclAllReduce.cpp
	mkdir -p bin
	$(HIPCC) -I$(RCCL_INC) -I$(TEST_INC) $(ARCHS) rcclAllReduce.cpp -L$(RCCL_LIB) -lrccl -o ./bin/allreduce

reduce: rcclReduce.cpp
	mkdir -p bin
	$(HIPCC) -I$(RCCL_INC) -I$(TEST_INC) $(ARCHS) rcclReduce.cpp -L$(RCCL_LIB) -lrccl -o ./bin/reduce

multistream: rcclMultiStream.cpp
	mkdir -p bin
	$(HIPCC) -I$(RCCL_INC) -I$(TEST_INC) $(ARCHS) rcclMultiStream.cpp -L$(RCCL_LIB) -lrccl -o ./bin/multistream

clean:
	rm -rf ./bin
