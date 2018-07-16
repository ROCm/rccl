all: commInitAll

commInitAll:
	mkdir -p bin
	/opt/rocm/hip/bin/hipcc -I/opt/rocm/rccl/include -I../../src --amdgpu-target=gfx900 --amdgpu-target=gfx803 rcclCommInitAll.cpp -L/opt/rocm/rccl/lib -lrccl -o ./bin/rcclCommInitAll

clean:
	rm -rf ./bin
