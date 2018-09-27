RCCL_DIR=/opt/rocm/rccl
HIP_DIR=/opt/rocm/hip
all: commInitAll
ARCHS=--amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906

commInitAll:
	mkdir -p bin
	${HIP_DIR}/bin/hipcc -I${RCCL_DIR}/include -I../ ${ARCHS} rcclCommInitAll.cpp -L${RCCL_DIR}/lib -lrccl -o ./bin/rcclCommInitAll

clean:
	rm -rf ./bin
