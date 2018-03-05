all: rcclKernelCopy

INC1=../
INC_SRC=../../../src
INC_INC=../../../inc

rcclKernelCopy:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I${INC1} -I${INC_SRC} -I${INC_INC} --amdgpu-target=gfx900 rcclKernelCopy.cpp -o ./bin/rcclKernelCopy

clean:
	rm -rf ./bin
