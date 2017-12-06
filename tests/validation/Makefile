all: bcast allreduce

bcast:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I/opt/rocm/rccl/include --amdgpu-target=gfx900 --amdgpu-target=gfx803 rcclBcast.cpp -L/opt/rocm/rccl/lib -lrccl -o ./bin/bcast

comm:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I/opt/rocm/rccl/include --amdgpu-target=gfx900 --amdgpu-target=gfx803 rcclCommBcast.cpp -L/opt/rocm/rccl/lib -lrccl -o ./bin/commBcast

allgather:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I/opt/rocm/rccl/include --amdgpu-target=gfx900 --amdgpu-target=gfx803 rcclAllGather.cpp -L/opt/rocm/rccl/lib -lrccl -o ./bin/allgather

allreduce:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I/opt/rocm/rccl/include --amdgpu-target=gfx900 --amdgpu-target=gfx803 rcclAllReduce.cpp -L/opt/rocm/rccl/lib -lrccl -o ./bin/allreduce

reduce:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I/opt/rocm/rccl/include --amdgpu-target=gfx900 --amdgpu-target=gfx803 rcclReduce.cpp -L/opt/rocm/rccl/lib -lrccl -o ./bin/reduce



clean:
	rm -rf ./bin
