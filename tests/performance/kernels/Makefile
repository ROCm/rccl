all: rcclKernelCopy rcclKernelReduceSum

INC1=../
INC_SRC=../../../src
INC_INC=../../../inc
INC_COMMON=../../

rcclKernelCopy:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I${INC1} -I${INC_SRC} -I${INC_INC} -I${INC_COMMON} --amdgpu-target=gfx900 rcclKernelCopy.cpp -o ./bin/rcclKernelCopy

rcclKernelReduceSum:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I${INC1} -I${INC_SRC} -I${INC_INC} -I${INC_COMMON} --amdgpu-target=gfx900 rcclKernelReduceSum.cpp -o ./bin/rcclKernelReduceSum

rcclBenchKernelCopy:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I${INC1} -I${INC_SRC} -I${INC_INC} -I${INC_COMMON} --amdgpu-target=gfx900 rcclBenchKernelCopy.cpp -o ./bin/rcclBenchKernelCopy

clean:
	rm -rf ./bin
