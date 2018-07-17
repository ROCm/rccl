HIP_DIR=/opt/rocm/hip
RCCL_DIR=/opt/rocm/rccl

all: allreduce reducescatter reduce

bcast:
	mkdir -p bin
	${HIP_DIR}/bin/hipcc -I${RCCL_DIR}/include --amdgpu-target=gfx900 rcclBcast.cpp -L${RCCL_DIR}/lib -lrccl -o ./bin/bcast

allreduce:
	mkdir -p bin
	${HIP_DIR}/bin/hipcc -I${RCCL_DIR}/include -I../ --amdgpu-target=gfx900 rcclAllReduce.cpp -L${RCCL_DIR}/lib -lrccl -o ./bin/allreduce

allgather:
	mkdir -p bin
	${HIP_DIR}/bin/hipcc -I${RCCL_DIR}/include -I../ --amdgpu-target=gfx900 rcclAllGather.cpp -L${RCCL_DIR}/lib -lrccl -o ./bin/allgather

reduce:
	mkdir -p bin
	${HIP_DIR}/bin/hipcc -I${RCCL_DIR}/include -I../ --amdgpu-target=gfx900 rcclReduce.cpp -L${RCCL_DIR}/lib -lrccl -o ./bin/reduce

reducescatter:
	mkdir -p bin
	${HIP_DIR}/bin/hipcc -I${RCCL_DIR}/include -I../ --amdgpu-target=gfx900 rcclReduceScatter.cpp -L${RCCL_DIR}/lib -lrccl -o ./bin/reducescatter

clean:
	rm -rf ./bin
