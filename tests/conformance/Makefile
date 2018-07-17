RCCL_DIR=/opt/rocm/rccl
HIP_DIR=/opt/rocm/hip
all: commInitAll

commInitAll:
	mkdir -p bin
	${HIP_DIR}/bin/hipcc -I${RCCL_DIR}/include -I../ --amdgpu-target=gfx900 --amdgpu-target=gfx803 rcclCommInitAll.cpp -L${RCCL_DIR}/lib -lrccl -o ./bin/rcclCommInitAll

clean:
	rm -rf ./bin
