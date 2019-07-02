/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * @file rccl_prim_test.cpp
 *
 * test performance if individual rccl primitives
 */
#include <cstdio>  //fprintf
#include <iostream> //cerr
#include <unistd.h> //usleep
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include "copy_kernel.h"

#define MAX_WORKGROUPS 8
#define THREADS 256

#define COPY_UNROLL       4
#define REDUCE_UNROLL     2
#define DOUBLECOPY_UNROLL 2
#define REDUCECOPY_UNROLL 2

struct transfer_data_t {
  float *dest0; //remote fine grain
  float *src0;  //local fine grain
  float *dest1; //local coarse grain
  float *src1;  //local coarse grain
  int N;
  int gpu;
};

struct profiling_data_t {
  uint64_t write_cycles;
  uint64_t bytes_transferred;
};


#define LOAD(VAR) __atomic_load_n((VAR), __ATOMIC_SEQ_CST)
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)

enum Ops {
  OP_COPY,
  OP_LOCALCOPY,
  OP_DOUBLECOPY,
  OP_REDUCE,
  OP_REDUCECOPY,
  NUM_OPS,
};

template<int op>
__global__ void flag_sync_kernel(struct transfer_data_t* transfer_data, struct profiling_data_t* profiling_data) {
  size_t idx = threadIdx.x;
  uint64_t curr_time, next_time;

  if (idx == 0) {
    curr_time = clock64();
  }

  int offset = transfer_data->N * blockIdx.x / gridDim.x;
  int n = transfer_data->N / gridDim.x;
  if (op == OP_COPY) Copy<COPY_UNROLL, THREADS, float>(transfer_data->dest0 + offset, transfer_data->src0 + offset, n);
  if (op == OP_LOCALCOPY) Copy<COPY_UNROLL, THREADS, float>(transfer_data->dest1 + offset, transfer_data->src0 + offset, n);
  if (op == OP_DOUBLECOPY) DoubleCopy<DOUBLECOPY_UNROLL, THREADS, float>(transfer_data->dest0 + offset, transfer_data->dest1 + offset, transfer_data->src0 + offset, n);
  if (op == OP_REDUCE) Reduce<REDUCE_UNROLL, THREADS, float>(transfer_data->dest0 + offset, transfer_data->src0 + offset, transfer_data->src1 + offset, n);
  if (op == OP_REDUCECOPY) ReduceCopy<REDUCECOPY_UNROLL, THREADS, float>(transfer_data->dest0 + offset, transfer_data->dest1 + offset, transfer_data->src0 + offset, transfer_data->src1 + offset, n);

  __syncthreads();
  if (idx == 0) {
    next_time = clock64();
    __atomic_fetch_add(&(profiling_data->write_cycles), next_time - curr_time, __ATOMIC_SEQ_CST);
    __atomic_fetch_add(&(profiling_data->bytes_transferred), n * sizeof(float), __ATOMIC_SEQ_CST);
  }
}

typedef void(*flag_sync_kernel_t)(struct transfer_data_t* transfer_data, struct profiling_data_t* profiling_data);

static flag_sync_kernel_t const flagSyncKerns[NUM_OPS] = {
  flag_sync_kernel<OP_COPY>,
  flag_sync_kernel<OP_LOCALCOPY>,
  flag_sync_kernel<OP_DOUBLECOPY>,
  flag_sync_kernel<OP_REDUCE>,
  flag_sync_kernel<OP_REDUCECOPY>,
};

__global__ void initTestDataKernel(float* data, const size_t N, const int gpu) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < N) {
    data[tid] = 1.0/(float)(gpu*17 + tid%77);
    tid += blockDim.x * gridDim.x;
  }
}

#define HIPCHECK(cmd)                                                          \
do {                                                                           \
  hipError_t error = (cmd);                                                    \
  if (error != hipSuccess)                                                     \
  {                                                                            \
    std::cerr << "Encountered HIP error (" << error << ") at line "            \
              << __LINE__ << " in file " << __FILE__ << "\n";                  \
    exit(-1);                                                                  \
  }                                                                            \
} while (0)

static void setupPeers() {
    int deviceCnt, dev;

     HIPCHECK(hipGetDeviceCount(&deviceCnt));
     HIPCHECK(hipGetDevice(&dev));
    //! If gpus are not peer enabled, enable them
    for (int i = 0; i < deviceCnt; i++) {
         HIPCHECK(hipSetDevice(i));
        for (int j = 0; j < deviceCnt; j++) {
            if (i != j) {
                HIPCHECK(hipDeviceEnablePeerAccess(j, 0));
            }
        }
    }
     HIPCHECK(hipSetDevice(dev));
}

char* getCmdOption(char ** begin, char ** end, const std::string & option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

int main(int argc,char* argv[])
{
  if (cmdOptionExists(argv, argv + argc, "-h")) {
    printf("./rccl_prim_test -w num_workgroups -p copy|localcopy|doublecopy|reduce|reducecopy|all -n iterations\n");
    exit(0);
  }

  int workgroups = 1;
  char *wg = getCmdOption(argv, argv + argc, "-w");
  if (wg)
    workgroups = atol(wg);
  printf("Benchmarking using %d workgroups\n", workgroups);

  int iters = 10;
  char *it = getCmdOption(argv, argv + argc, "-n");
  if (it)
    iters = atol(it);
  printf("Benchmarking using %d iterations\n", iters);

  const char *ops[] = {"copy", "localcopy", "doublecopy", "reduce", "reducecopy", "all"};
  char *prim = getCmdOption(argv, argv + argc, "-p");
  int op = 5, begin_op, end_op;
  if (prim) {
    for (op = 0; op < sizeof(ops); op++)
      if (!strcmp((const char *)prim, ops[op]))
        break;
  }
  if (op < NUM_OPS ) {
    begin_op = op;
    end_op = op + 1;
  } else {
    begin_op = 0;
    end_op = NUM_OPS;
    printf("Benchmarking all ops\n");
  }

  // Enable peer access
  setupPeers();

  // data buffers
  float *buff_0, *buff_1, *buff_coarse_0, *buff_coarse_1;
  struct transfer_data_t h_transfer_data_0, h_transfer_data_1, *transfer_data_0, *transfer_data_1;
  struct profiling_data_t *profiling_data_0, *profiling_data_1, *d_profiling_data_0, *d_profiling_data_1;
  uint64_t N = 2097152*4*MAX_WORKGROUPS;

  int hipDev = 0;
  HIPCHECK(hipSetDevice(hipDev));
  hipDeviceProp_t prop;
  HIPCHECK(hipGetDeviceProperties(&prop, hipDev));
  printf("#   device %d [0x%02x] %s\n",
                  hipDev, prop.pciBusID, prop.name);
  HIPCHECK(hipExtMallocWithFlags((void**) &transfer_data_0, sizeof(struct transfer_data_t), hipDeviceMallocFinegrained));
  //printf("GPU 0: allocated fine grain VRAM at %llx\n", (unsigned long long)transfer_data_0);
  HIPCHECK(hipExtMallocWithFlags((void**) &buff_0, 2*N*sizeof(float), hipDeviceMallocFinegrained));
  //printf("GPU 0: allocated fine grain VRAM at %llx\n", (unsigned long long)buff_0);
  HIPCHECK(hipMalloc((void**) &buff_coarse_0, 2*N*sizeof(float)));
  //printf("GPU 0: allocated coarse grain VRAM at %llx\n", (unsigned long long)buff_coarse_0);
  profiling_data_0 = (struct profiling_data_t *)malloc(sizeof(struct profiling_data_t));
  HIPCHECK(hipMalloc((void**) &d_profiling_data_0, sizeof(struct profiling_data_t)));
  //create stream
  hipStream_t stream_0;
  HIPCHECK(hipStreamCreate(&stream_0));
  //randomize test data
  hipLaunchKernelGGL(initTestDataKernel,
      /*grid dim x,y,z*/        dim3(32, 1, 1),
      /*block dim x,y,z*/       dim3(THREADS, 1, 1),
      /*dynamic shared mem*/    0,
      /*stream*/                stream_0,
      /*kernel args*/           buff_0, 2*N, 0);
  hipLaunchKernelGGL(initTestDataKernel,
      /*grid dim x,y,z*/        dim3(32, 1, 1),
      /*block dim x,y,z*/       dim3(THREADS, 1, 1),
      /*dynamic shared mem*/    0,
      /*stream*/                stream_0,
      /*kernel args*/           buff_coarse_0, 2*N, 0);

  hipDev = 1;
  HIPCHECK(hipSetDevice(hipDev));
  HIPCHECK(hipGetDeviceProperties(&prop, hipDev));
  printf("#   device %d [0x%02x] %s\n",
                  hipDev, prop.pciBusID, prop.name);
  HIPCHECK(hipExtMallocWithFlags((void**) &transfer_data_1, sizeof(struct transfer_data_t), hipDeviceMallocFinegrained));
  //printf("GPU 1: allocated fine grain VRAM at %llx\n", (unsigned long long)transfer_data_1);
  HIPCHECK(hipExtMallocWithFlags((void**) &buff_1, 2*N*sizeof(float), hipDeviceMallocFinegrained));
  //printf("GPU 1: allocated fine grain VRAM at %llx\n", (unsigned long long)buff_1);
  HIPCHECK(hipMalloc((void**) &buff_coarse_1, 2*N*sizeof(float)));
  //printf("GPU 1: allocated coarse grain VRAM at %llx\n", (unsigned long long)buff_coarse_1);
  profiling_data_1 = (struct profiling_data_t *)malloc(sizeof(struct profiling_data_t));
  HIPCHECK(hipMalloc((void**) &d_profiling_data_1, sizeof(struct profiling_data_t)));
  //create stream
  hipStream_t stream_1;
  HIPCHECK(hipStreamCreate(&stream_1));
  //randomize test data
  hipLaunchKernelGGL(initTestDataKernel,
      /*grid dim x,y,z*/        dim3(32, 1, 1),
      /*block dim x,y,z*/       dim3(THREADS, 1, 1),
      /*dynamic shared mem*/    0,
      /*stream*/                stream_1,
      /*kernel args*/           buff_1, 2*N, 1);
  hipLaunchKernelGGL(initTestDataKernel,
      /*grid dim x,y,z*/        dim3(32, 1, 1),
      /*block dim x,y,z*/       dim3(THREADS, 1, 1),
      /*dynamic shared mem*/    0,
      /*stream*/                stream_1,
      /*kernel args*/           buff_coarse_1, 2*N, 1);

  h_transfer_data_0.dest0 = buff_1;
  h_transfer_data_0.dest1 = buff_coarse_0 + N;
  h_transfer_data_0.src0 = buff_0;
  h_transfer_data_0.src1 = buff_coarse_0;
  h_transfer_data_0.N = N;
  h_transfer_data_0.gpu = 0;

  h_transfer_data_1.dest0 = buff_0 + N;
  h_transfer_data_1.dest1 = buff_coarse_1;
  h_transfer_data_1.src0 = buff_1 + N;
  h_transfer_data_1.src1 = buff_coarse_1 + N;
  h_transfer_data_1.N = N;
  h_transfer_data_1.gpu = 1;

  HIPCHECK(hipSetDevice(0));
  HIPCHECK(hipMemcpyAsync(transfer_data_0, &h_transfer_data_0,
                          sizeof(struct transfer_data_t), hipMemcpyHostToDevice,
                          stream_0));
  HIPCHECK(hipStreamSynchronize(stream_0));

  HIPCHECK(hipSetDevice(1));
  HIPCHECK(hipMemcpyAsync(transfer_data_1, &h_transfer_data_1,
                          sizeof(struct transfer_data_t), hipMemcpyHostToDevice,
                          stream_1));
  HIPCHECK(hipStreamSynchronize(stream_1));

  for (int op = begin_op; op < end_op; op ++) {
    const char *OpsName[] = {"Copy", "Local Copy", "Double Copy", "Reduce", "ReduceCopy"};
    printf("Testing %s: \n", OpsName[op]);
    // 2 warm up cycles
    for (int i = 0; i < 2; i ++) {
      HIPCHECK(hipSetDevice(0));
      //launch the kernel
      hipLaunchKernelGGL(flagSyncKerns[op],
          /*grid dim x,y,z*/        dim3(workgroups, 1, 1),
          /*block dim x,y,z*/       dim3(THREADS, 1, 1),
          /*dynamic shared mem*/    0,
          /*stream*/                stream_0,
          /*kernel args*/           transfer_data_0, d_profiling_data_0);

      HIPCHECK(hipSetDevice(1));
      //launch the kernel
      hipLaunchKernelGGL(flagSyncKerns[op],
          /*grid dim x,y,z*/        dim3(workgroups, 1, 1),
          /*block dim x,y,z*/       dim3(THREADS, 1, 1),
          /*dynamic shared mem*/    0,
          /*stream*/                stream_1,
          /*kernel args*/           transfer_data_1, d_profiling_data_1);
    }

    HIPCHECK(hipSetDevice(0));
    HIPCHECK(hipStreamSynchronize(stream_0));
    HIPCHECK(hipMemset(d_profiling_data_0, 0, sizeof(struct profiling_data_t)));
    HIPCHECK(hipSetDevice(1));
    HIPCHECK(hipStreamSynchronize(stream_1));
    HIPCHECK(hipMemset(d_profiling_data_1, 0, sizeof(struct profiling_data_t)));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i ++) {
      HIPCHECK(hipSetDevice(0));
      //launch the kernel
      hipLaunchKernelGGL(flagSyncKerns[op],
          /*grid dim x,y,z*/        dim3(workgroups, 1, 1),
          /*block dim x,y,z*/       dim3(THREADS, 1, 1),
          /*dynamic shared mem*/    0,
          /*stream*/                stream_0,
          /*kernel args*/           transfer_data_0, d_profiling_data_0);

      HIPCHECK(hipSetDevice(1));
      //launch the kernel
      hipLaunchKernelGGL(flagSyncKerns[op],
          /*grid dim x,y,z*/        dim3(workgroups, 1, 1),
          /*block dim x,y,z*/       dim3(THREADS, 1, 1),
          /*dynamic shared mem*/    0,
          /*stream*/                stream_1,
          /*kernel args*/           transfer_data_1, d_profiling_data_1);
    }

    HIPCHECK(hipSetDevice(0));
    HIPCHECK(hipStreamSynchronize(stream_0));
    HIPCHECK(hipSetDevice(1));
    HIPCHECK(hipStreamSynchronize(stream_1));
    auto delta = std::chrono::high_resolution_clock::now() - start;
    double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();

    HIPCHECK(hipMemcpyAsync(profiling_data_0, d_profiling_data_0,
                            sizeof(struct profiling_data_t), hipMemcpyDeviceToHost,
                            stream_0));
    HIPCHECK(hipStreamSynchronize(stream_0));

    HIPCHECK(hipMemcpyAsync(profiling_data_1, d_profiling_data_1,
                            sizeof(struct profiling_data_t), hipMemcpyDeviceToHost,
                            stream_1));
    HIPCHECK(hipStreamSynchronize(stream_1));

    double speed = (double)(profiling_data_0->bytes_transferred) / (deltaSec*1.0E9);
    printf("Transfered %lu bytes in %f s. Throughput %f GB/s\n", profiling_data_0->bytes_transferred, deltaSec, speed);

#define RTC_CLOCK_FREQ 2.7E07
    double t0 = (double)profiling_data_0->write_cycles/((double)RTC_CLOCK_FREQ)/(double)workgroups;
    fprintf(stderr, "GPU 0: time %.4fs bytes_transferred %lu kernel throughput %.2f GB/s\n",
      t0, profiling_data_0->bytes_transferred, (double)profiling_data_0->bytes_transferred/(t0*1.0E9));

    double t1 = (double)profiling_data_1->write_cycles/((double)RTC_CLOCK_FREQ)/(double)workgroups;
    fprintf(stderr, "GPU 1: time %.4fs bytes_transferred %lu kernel throughput %.2f GB/s\n",
      t1, profiling_data_1->bytes_transferred, (double)profiling_data_0->bytes_transferred/(t1*1.0E9));
  }

  HIPCHECK(hipStreamDestroy(stream_0));
  HIPCHECK(hipStreamDestroy(stream_1));
  HIPCHECK(hipFree((void*) transfer_data_0));
  HIPCHECK(hipFree((void*) buff_0));
  HIPCHECK(hipFree((void*) buff_coarse_0));
  HIPCHECK(hipFree((void*) d_profiling_data_0));
  free(profiling_data_0);
  HIPCHECK(hipFree((void*) transfer_data_1));
  HIPCHECK(hipFree((void*) buff_1));
  HIPCHECK(hipFree((void*) buff_coarse_1));
  HIPCHECK(hipFree((void*) d_profiling_data_1));
  free(profiling_data_1);
}
