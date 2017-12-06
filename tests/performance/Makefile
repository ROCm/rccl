all: bcast

bcast:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I/opt/rocm/rccl/include --amdgpu-target=gfx900 rcclBcast.cpp -L/opt/rocm/rccl/lib -lrccl -o ./bin/bcast

allreduce:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I/opt/rocm/rccl/include --amdgpu-target=gfx900 rcclAllReduce.cpp -L/opt/rocm/rccl/lib -lrccl -o ./bin/allreduce

allgather:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I/opt/rocm/rccl/include --amdgpu-target=gfx900 rcclAllGather.cpp -L/opt/rocm/rccl/lib -lrccl -o ./bin/allgather


clean:
	rm -rf ./bin
