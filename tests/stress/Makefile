all: reduce

ROCM_PATH=/opt/rocm
TEST_INC=../
RCCL_INC=$(ROCM_PATH)/rccl/include
RCCL_LIB=$(ROCM_PATH)/rccl/lib
HIPCC=$(ROCM_PATH)/bin/hipcc
ARCHS=--amdgpu-target=gfx900 --amdgpu-target=gfx803 --amdgpu-target=gfx906

bcast:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I$(RCCL_INC) -I$(TEST_INC) $(ARCHS) rcclBcast.cpp -L$(RCCL_LIB) -lrccl -o ./bin/bcast

comm:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I$(RCCL_INC) -I$(TEST_INC) $(ARCHS) rcclCommBcast.cpp -L$(RCCL_LIB) -lrccl -o ./bin/commBcast

allgather:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I$(RCCL_INC) -I$(TEST_INC) $(ARCHS) rcclAllGather.cpp -L$(RCCL_LIB) -lrccl -o ./bin/allgather

allreduce:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I$(RCCL_INC) -I$(TEST_INC) $(ARCHS) rcclAllReduce.cpp -L$(RCCL_LIB) -lrccl -o ./bin/allreduce

reduce:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I$(RCCL_INC) -I$(TEST_INC) $(ARCHS) rcclReduce.cpp -L$(RCCL_LIB) -lrccl -o ./bin/reduce

reducescatter:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I$(RCCL_INC) -I$(TEST_INC) $(ARCHS) rcclReduceScatter.cpp -L$(RCCL_LIB) -lrccl -o ./bin/reducescatter


clean:
	rm -rf ./bin
