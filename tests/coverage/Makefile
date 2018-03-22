all: allreduce

allreduce:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I/opt/rocm/rccl/include --amdgpu-target=gfx900 --amdgpu-target=gfx803 rcclAllReduce.cpp -L/opt/rocm/rccl/lib -lrccl -o ./bin/allreduce


clean:
	rm -rf ./bin
