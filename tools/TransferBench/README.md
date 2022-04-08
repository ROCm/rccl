# TransferBench

TransferBench is a simple utility capable of benchmarking simultaneous copies between user-specified devices (CPUs/GPUs).

## Requirements

1. ROCm stack installed on the system (HIP runtime)
2. libnuma installed on system

## Building
  To build TransferBench:
* `make`

  If ROCm is installed in a folder other than `/opt/rocm/`, set ROCM_PATH appropriately
