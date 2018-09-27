HIP_DIR=/opt/rocm/hip
RCCL_DIR=/opt/rocm/rccl
ARCHS=--amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906
all: allreduce reduce

bcast: rcclBcast.cpp
	mkdir -p bin
	${HIP_DIR}/bin/hipcc -I${RCCL_DIR}/include -I../ ${ARCHS} rcclBcast.cpp -L${RCCL_DIR}/lib -lrccl -o ./bin/bcast

allreduce: rcclAllReduce.cpp
	mkdir -p bin
	${HIP_DIR}/bin/hipcc -I${RCCL_DIR}/include -I../ ${ARCHS} rcclAllReduce.cpp -L${RCCL_DIR}/lib -lrccl -o ./bin/allreduce

reduce: rcclReduce.cpp
	mkdir -p bin
	${HIP_DIR}/bin/hipcc -I${RCCL_DIR}/include -I../ ${ARCHS} rcclReduce.cpp -L${RCCL_DIR}/lib -lrccl -o ./bin/reduce

clean:
	rm -rf ./bin
