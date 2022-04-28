/*
Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <vector>
#include <sstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <set>
#include <unistd.h>
#include <map>
#include <iostream>
#include <sstream>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hsa/hsa_ext_amd.h>

#include "EnvVars.hpp"

// Helper macro for catching HIP errors
#define HIP_CALL(cmd)                                                   \
    do {                                                                \
        hipError_t error = (cmd);                                       \
        if (error != hipSuccess)                                        \
        {                                                               \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " \
                      << __LINE__ << " in file " << __FILE__ << "\n";   \
            exit(-1);                                                   \
        }                                                               \
    } while (0)

// Simple configuration parameters
size_t const DEFAULT_BYTES_PER_TRANSFER = (1<<26);  // Amount of data transferred per Transfer

// Different src/dst memory types supported
typedef enum
{
  MEM_CPU      = 0,    // Coarse-grained pinned CPU memory
  MEM_GPU      = 1,    // Coarse-grained global GPU memory
  MEM_CPU_FINE = 2,    // Fine-grained pinned CPU memory
  MEM_GPU_FINE = 3     // Fine-grained global GPU memory
} MemType;

char const MemTypeStr[5] = "CGBF";

typedef enum
{
  MODE_FILL  = 0,         // Fill data with pattern
  MODE_CHECK = 1          // Check data against pattern
} ModeType;

// Each threadblock copies N floats from src to dst
struct BlockParam
{
  int       N;
  float*    src;
  float*    dst;
  long long startCycle;
  long long stopCycle;
};

// Each Transfer is a uni-direction operation from a src memory to dst memory
struct Transfer
{
  int     transferIndex;       // Transfer identifier

  // Transfer config
  MemType exeMemType;          // Transfer executor type (CPU or GPU)
  int     exeIndex;            // Executor index (NUMA node for CPU / device ID for GPU)
  MemType srcMemType;          // Source memory type
  int     srcIndex;            // Source device index
  MemType dstMemType;          // Destination memory type
  int     dstIndex;            // Destination device index
  int     numBlocksToUse;      // Number of threadblocks to use for this Transfer

  // Memory
  float*  srcMem;              // Source memory
  float*  dstMem;              // Destination memory

  // How memory is split across threadblocks / CPU cores
  std::vector<BlockParam> blockParam;
  BlockParam* blockParamGpuPtr;

  // Results
  double  transferTime;

  // Prepares src memory and how to divide N elements across threadblocks/threads
  void PrepareBlockParams(EnvVars const& ev, size_t const N);
};

typedef std::pair<MemType, int> Executor;

struct ExecutorInfo
{
  std::vector<Transfer>    transfers;     // Transfers to execute

  // For GPU-Executors
  int                      totalBlocks;   // Total number of CUs/CPU threads to use
  BlockParam*              blockParamGpu; // Copy of block parameters in GPU device memory
  std::vector<hipStream_t> streams;
  std::vector<hipEvent_t>  startEvents;
  std::vector<hipEvent_t>  stopEvents;

  // Results
  double totalTime;
};

typedef std::map<Executor, ExecutorInfo> TransferMap;

// Display usage instructions
void DisplayUsage(char const* cmdName);

// Display detected GPU topology / CPU numa nodes
void DisplayTopology(bool const outputToCsv);

// Build array of test sizes based on sampling factor
void PopulateTestSizes(size_t const numBytesPerTransfer, int const samplingFactor,
                       std::vector<size_t>& valuesofN);

void ParseMemType(std::string const& token, int const numCpus, int const numGpus,
                  MemType* memType, int* memIndex);

void ParseTransfers(char* line, int numCpus, int numGpus,
                TransferMap& transferMap);

void EnablePeerAccess(int const deviceId, int const peerDeviceId);
void AllocateMemory(MemType memType, int devIndex, size_t numBytes, void** memPtr);
void DeallocateMemory(MemType memType, void* memPtr);
void CheckPages(char* byteArray, size_t numBytes, int targetId);
void CheckOrFill(ModeType mode, int N, bool isMemset, bool isHipCall, std::vector<float> const& fillPattern, float* ptr);
void RunTransfer(EnvVars const& ev, size_t const N, int const iteration, ExecutorInfo& exeInfo, int const transferIdx);
void RunPeerToPeerBenchmarks(EnvVars const& ev, size_t N, int numBlocksToUse, int readMode, int skipCpu);

// Return the maximum bandwidth measured for given (src/dst) pair
double GetPeakBandwidth(EnvVars const& ev,
                        size_t  const  N,
                        int     const  isBidirectional,
                        int     const  readMode,
                        int     const  numBlocksToUse,
                        MemType const  srcMemType,
                        int     const  srcIndex,
                        MemType const  dstMemType,
                        int     const  dstIndex);

std::string GetLinkTypeDesc(uint32_t linkType, uint32_t hopCount);
std::string GetDesc(MemType srcMemType, int srcIndex,
                    MemType dstMemType, int dstIndex);
std::string GetTransferDesc(Transfer const& transfer);
int RemappedIndex(int const origIdx, MemType const memType);
int GetWallClockRate(int deviceId);
