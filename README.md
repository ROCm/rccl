# RCCL
ROCm Communication Collectives Library

## Introduction
RCCL (rickle) is implementation of MPI communication apis on ROCm enabled GPUs. It is a collective communication library whose aim is to provide low-latency and high-bandwidth communication on dense GPU systems. RCCL launches special-purpose compute kernels for parallel overlapping transfers. This involves distributed processing and exchanging data between participating peer-accessible GPUs in a logical ring within a single multi-GPU node. 

## Supported APIs
1. AllReduce
2. Broadcast

## Requirements
1. ROCm supported GPUs
2. ROCm stack installed on the system (HCC)

## Build
RCCL directly depends on HIP runtime & HCC C++ compiler which are part of the ROCm SW stack.
```bash
git clone https://github.com/ROCmSoftwarePlatform/rccl.git
cd rccl/src
make lib
sudo make install
```
RCCL install requires sudo/root access because it creates a directory called "rccl" under /opt/rocm/. This is an optional step and RCCL can be used directly by including the path containing librccl.so.

## Run
`rccl` library install directory should be added to `LD_LIBRARY_PATH`
```bash
export LD_LIBRARY_PATH=/opt/rocm/rccl/lib:$LD_LIBRARY_PATH
```

## Usage
```cpp
#include <rccl.h>
#include <vector>

int main() {
  int numGpus;
  hipGetDeviceCount(&numGpus);
  std::vector<rcclComm_t> comms(numGpus);
  rcclCommInitAll(comms, numGpus);

  std::vector<float*> sendBuff(numGpus);
  std::vector<float*> recvBuff(numGpus);

  std::vector<hipStream_t> streams(numGpus);

  // Set up sendBuff and recvBuff on each GPU
  // Create stream on each GPU

  for(int i=0;i<numGpus;i++) {
    hipSetDevice(i);
    rcclAllReduce(sendBuff[i], recvBuff[i], size, rcclFloat,
      rcclSum, comms[i], streams[i]);
  }

}
```

## Source Layout
* `inc` - contains the public RCCL header exposing the RCCL interfaces
* `src` - contains source code for the implementation of the RCCL APIs
* `tests` - contains unit tests cases to validate RCCL

## Source Naming
The RCCL library consists of two primary layers:

### Interface layer
* rccl.h - C99 APIs as defined by the RCCL library.
* rccl.cpp - The interface layer implementation encapsulates the functionality by invoking the actual primitive specific C++ template functions.


### RCCL primitive specific implementations and kernels
* rcclDataTypes.h
* rcclTracker.h
* rccl{Primitive}Runtime.h
* rccl{Primitive}Kernels.h
* rcclKernelHelper.h

This communication primitive layer contains implementations the broadcast and all-reduce distributed designs. It takes into account of the number of GPUs in the system, the data size and divide the work into smaller chunks. The compute kernels are launched to implement a distributed pipelined operation that achieves overall maximum throughput when compared to a single root processing and sharing the data serially.

## Caveats
The initial implementation of the distributed broadcast and all-reduce designs are focused on the functionality and correctness and not tuned yet to obtain optimal performance for a specific input size and GPU count. Better strategies to determine optimal chunk sizes to allow overlapping of transfers for better pipelining are being explored.
