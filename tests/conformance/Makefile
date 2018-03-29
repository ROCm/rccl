all: commInitAll

commInitAll:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I/opt/rocm/rccl/include --amdgpu-target=gfx900 --amdgpu-target=gfx803 rcclCommInitAll.cpp -L/opt/rocm/rccl/lib -lrccl -o ./bin/rcclCommInitAll

clean:
	rm -rf ./bin
