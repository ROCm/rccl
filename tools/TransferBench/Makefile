HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=TransferBench
CXXFLAGS = -O3 -fopenmp -I../../src/include -I.

all: $(EXE)

$(EXE): $(EXE).cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o $(EXE)
