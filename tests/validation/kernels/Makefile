all: rcclKernelCopy
 
INC1=../
INC_SRC=../../../src
INC_INC=../../../inc
INC_COMMON=../../

rcclKernelCopy:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I${INC1} -I${INC_SRC} -I${INC_INC} -I${INC_COMMON} --amdgpu-target=gfx900 rcclKernelCopy.cpp -o ./bin/rcclKernelCopy

rcclValidateKernelCopy:
	mkdir -p bin
	/opt/rocm/bin/hipcc -I${INC1} -I${INC_SRC} -I${INC_INC} -I${INC_COMMON} --amdgpu-target=gfx900 rcclValidateKernelCopy.cpp -o ./bin/rcclValidateKernelCopy

clean:
	rm -rf ./bin
