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

#define MAX_GPU 8
#define MAX_WORKGROUPS 8
#define THREADS 256

#define COPY_UNROLL       4
#define REDUCE_UNROLL     2
#define DOUBLECOPY_UNROLL 2
#define REDUCECOPY_UNROLL 2

struct transfer_data_t {
  float *dest0[MAX_WORKGROUPS]; //remote fine grain
  float *src0[MAX_WORKGROUPS];  //local fine grain
  float *dest1[MAX_WORKGROUPS]; //local coarse grain
  float *src1[MAX_WORKGROUPS];  //local coarse grain
  int N;
  int gpu;
  int ngpu;
  uint64_t *remOpCount;
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

template<int op, int sync>
__global__ void flag_sync_kernel(struct transfer_data_t* transfer_data, struct profiling_data_t* profiling_data, uint64_t opCount) {
  size_t idx = threadIdx.x;
  uint64_t curr_time, next_time;
  int bid = blockIdx.x;
  int n = transfer_data->N;

  // signal self ready and wait until all GPUs are ready
  if (idx == 0) {
    if (bid == 0)
      STORE(&transfer_data->remOpCount[transfer_data->gpu], opCount);
    if (sync) {
      for (int i = 0; i < transfer_data->ngpu; i++) {
        while (LOAD(&transfer_data->remOpCount[i]) < opCount) {};
      }
    }
  }
  __syncthreads();

  if (idx == 0) {
    curr_time = clock64();
  }

  if (op == OP_COPY) Copy<COPY_UNROLL, THREADS, float>(transfer_data->dest0[bid], transfer_data->src0[bid], n);
  if (op == OP_LOCALCOPY) Copy<COPY_UNROLL, THREADS, float>(transfer_data->dest1[bid], transfer_data->src0[bid], n);
  if (op == OP_DOUBLECOPY) DoubleCopy<DOUBLECOPY_UNROLL, THREADS, float>(transfer_data->dest0[bid], transfer_data->dest1[bid], transfer_data->src0[bid], n);
  if (op == OP_REDUCE) Reduce<REDUCE_UNROLL, THREADS, float>(transfer_data->dest0[bid], transfer_data->src0[bid], transfer_data->src1[bid], n);
  if (op == OP_REDUCECOPY) ReduceCopy<REDUCECOPY_UNROLL, THREADS, float>(transfer_data->dest0[bid], transfer_data->dest1[bid], transfer_data->src0[bid], transfer_data->src1[bid], n);

  __syncthreads();
  if (idx == 0) {
    next_time = clock64();
    __atomic_fetch_add(&(profiling_data->write_cycles), next_time - curr_time, __ATOMIC_SEQ_CST);
    __atomic_fetch_add(&(profiling_data->bytes_transferred), n * sizeof(float), __ATOMIC_SEQ_CST);
  }
}

typedef void(*flag_sync_kernel_t)(struct transfer_data_t* transfer_data, struct profiling_data_t* profiling_data, uint64_t opCount);

static flag_sync_kernel_t const flagSyncKerns[NUM_OPS*2] = {
  flag_sync_kernel<OP_COPY, 0>,
  flag_sync_kernel<OP_COPY, 1>,
  flag_sync_kernel<OP_LOCALCOPY, 0>,
  flag_sync_kernel<OP_LOCALCOPY, 1>,
  flag_sync_kernel<OP_DOUBLECOPY, 0>,
  flag_sync_kernel<OP_DOUBLECOPY, 1>,
  flag_sync_kernel<OP_REDUCE, 0>,
  flag_sync_kernel<OP_REDUCE, 1>,
  flag_sync_kernel<OP_REDUCECOPY, 0>,
  flag_sync_kernel<OP_REDUCECOPY, 1>,
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

static void setupPeers(uint32_t *info) {
  int deviceCnt, dev;

  HIPCHECK(hipGetDeviceCount(&deviceCnt));
  HIPCHECK(hipGetDevice(&dev));
  //! If gpus are not peer enabled, enable them
  for (int i = 0; i < deviceCnt; i++) {
    HIPCHECK(hipSetDevice(i));
    for (int j = 0; j < deviceCnt; j++) {
      if (i != j) {
	int p2p;
        HIPCHECK(hipDeviceCanAccessPeer(&p2p, i, j));
        if (!p2p) {
          printf("Cannot enable peer access between device %d and %d. You may use HIP_VISIBLE_DEVICES to limit GPUs.\n",
           i, j);
          exit(-1);
        }
        HIPCHECK(hipDeviceEnablePeerAccess(j, 0));
        uint32_t linktype;
        HIPCHECK(hipExtGetLinkTypeAndHopCount(i, j, &linktype, &info[i*deviceCnt+j]));
      }
      else
        info[i*deviceCnt+j] = 0;
    }
  }
  HIPCHECK(hipSetDevice(dev));
}

static void printRing(int id, int *ring, int deviceCnt) {
  printf("Ring %d: ", id);
  for (int i = 0; i < deviceCnt; i++)
    printf("%1d ", ring[i]);
  printf("\n");
}

static void findConnect(uint32_t *info, int *ring, int deviceCnt) {
  int n = 0, curr = 0, best;
  uint32_t temp[MAX_GPU*MAX_GPU];
  for (int i = 0; i < deviceCnt*deviceCnt; i++) temp[i] = 0;
  for (int i = 0; i < deviceCnt; i++) {
    for (int j = 0; j < deviceCnt; j++) temp[j*deviceCnt+curr] = 1;
    ring[n] = curr;
    n++;
    int hops = 99;
    for (int j = 0; j < deviceCnt; j++) {
      if (temp[curr*deviceCnt+j]) continue;
      if (info[curr*deviceCnt+j] < hops) {
        best = j;
        hops = info[curr*deviceCnt+j];
      }
    }
    curr = best;
  }
}

static int findNextGpu(int *ring, int gpu, int deviceCnt) {
  int i;
  for (i = 0; i < deviceCnt; i ++)
    if (ring[i] == gpu) break;
  return ring[(i+1)%deviceCnt];
}

static void setupRings(uint32_t *info, int *ring_0, int *ring_1) {
  int deviceCnt, dev;
  HIPCHECK(hipGetDeviceCount(&deviceCnt));
  printf("Connection matrix:\n");
  for (int i = 0; i < deviceCnt; i++) {
    for (int j = 0; j < deviceCnt; j++)
      printf("%2d ", info[i*deviceCnt+j]);
    printf("\n");
  }
  findConnect(info, ring_0, deviceCnt);
  printRing(0, ring_0, deviceCnt);
  ring_1[0] =0;
  for (int i = 1; i < deviceCnt; i++)
    ring_1[i] = ring_0[deviceCnt-i];
  printRing(1, ring_1, deviceCnt);
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
    printf("./rccl_prim_test -w num_workgroups -p copy|localcopy|doublecopy|reduce|reducecopy|all -i iterations -n bytes -s 0|1\n");
    exit(0);
  }

  int workgroups = 1;
  char *wg = getCmdOption(argv, argv + argc, "-w");
  if (wg)
    workgroups = atol(wg);
  printf("Benchmarking using %d workgroups\n", workgroups);

  int iters = 10;
  char *it = getCmdOption(argv, argv + argc, "-i");
  if (it)
    iters = atol(it);
  printf("Benchmarking using %d iterations\n", iters);

  uint64_t nBytes = 2097152;
  char *nb = getCmdOption(argv, argv + argc, "-n");
  if (nb)
    nBytes = atol(nb);
  printf("Benchmarking using %ld bytes\n", nBytes);
  uint64_t N = nBytes/sizeof(float);

  int sync = 0;
  char *s = getCmdOption(argv, argv + argc, "-s");
  if (s)
    sync = atol(s);
  if (sync) printf("Sync all GPUs before operation\n");

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

  uint32_t connection_info[MAX_GPU*MAX_GPU];
  // Enable peer access
  setupPeers(connection_info);
  // clockwise and counter clockwise rings
  int ring_0[MAX_GPU] = {-1, -1, -1, -1};
  int ring_1[MAX_GPU] = {-1, -1, -1, -1};
  setupRings(connection_info, ring_0, ring_1);

  // data buffers
  float *buff[MAX_GPU*MAX_WORKGROUPS], *buff_coarse[MAX_GPU*MAX_WORKGROUPS];
  struct transfer_data_t h_transfer_data[MAX_GPU], *transfer_data[MAX_GPU];
  struct profiling_data_t *profiling_data[MAX_GPU], *d_profiling_data[MAX_GPU];
  hipStream_t stream[MAX_GPU];

  int nGpu = 1;
  HIPCHECK(hipGetDeviceCount(&nGpu));
  uint64_t *remOpCount, *d_remOpCount;
  HIPCHECK(hipHostMalloc((void**)&remOpCount, sizeof(uint64_t)*MAX_GPU, hipHostMallocMapped));
  HIPCHECK(hipHostGetDevicePointer((void**)&d_remOpCount, (void*)remOpCount, 0));


  for (int i = 0; i < nGpu; i ++) {
    HIPCHECK(hipSetDevice(i));
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, i));
    printf("#   device %d [0x%02x] %s\n",
                    i, prop.pciBusID, prop.name);
    //create stream
    HIPCHECK(hipStreamCreate(&stream[i]));
    profiling_data[i] = (struct profiling_data_t *)malloc(sizeof(struct profiling_data_t));
    HIPCHECK(hipMalloc((void**) &d_profiling_data[i], sizeof(struct profiling_data_t)));

    HIPCHECK(hipExtMallocWithFlags((void**) &transfer_data[i], sizeof(struct transfer_data_t), hipDeviceMallocFinegrained));
    for (int j = 0; j < workgroups; j++) {
      HIPCHECK(hipExtMallocWithFlags((void**) &buff[i*MAX_WORKGROUPS+j], 2*N*sizeof(float), hipDeviceMallocFinegrained));
      HIPCHECK(hipMalloc((void**) &buff_coarse[i*MAX_WORKGROUPS+j], 2*N*sizeof(float)));
      //randomize test data
      hipLaunchKernelGGL(initTestDataKernel,
          /*grid dim x,y,z*/        dim3(32, 1, 1),
          /*block dim x,y,z*/       dim3(THREADS, 1, 1),
          /*dynamic shared mem*/    0,
          /*stream*/                stream[i],
          /*kernel args*/           buff[i*MAX_WORKGROUPS+j], 2*N, 0);
      hipLaunchKernelGGL(initTestDataKernel,
          /*grid dim x,y,z*/        dim3(32, 1, 1),
          /*block dim x,y,z*/       dim3(THREADS, 1, 1),
          /*dynamic shared mem*/    0,
          /*stream*/                stream[i],
          /*kernel args*/           buff_coarse[i*MAX_WORKGROUPS+j], 2*N, 0);
    }
  }

  for (int i = 0; i < nGpu; i ++) {
    for (int j = 0; j < workgroups; j++) {
      int next_gpu;
      if (j%2)
        next_gpu = findNextGpu(ring_1, i, nGpu);
      else
        next_gpu = findNextGpu(ring_0, i, nGpu);
      //printf("GPU %d Ring %d -> Next GPU %d\n", i, j, next_gpu);
      h_transfer_data[i].dest0[j] = buff[next_gpu*MAX_WORKGROUPS+j] + N;
      h_transfer_data[i].dest1[j] = buff_coarse[i*MAX_WORKGROUPS+j] + N;
      h_transfer_data[i].src0[j] = buff[i*MAX_WORKGROUPS+j];
      h_transfer_data[i].src1[j] = buff_coarse[i*MAX_WORKGROUPS+j];
    }
    h_transfer_data[i].N = N;
    h_transfer_data[i].gpu = i;
    h_transfer_data[i].ngpu = nGpu;
    h_transfer_data[i].remOpCount = d_remOpCount;
  }

  for (int i = 0; i < nGpu; i ++) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMemcpyAsync(transfer_data[i], &h_transfer_data[i],
                            sizeof(struct transfer_data_t), hipMemcpyHostToDevice,
                            stream[i]));
    HIPCHECK(hipStreamSynchronize(stream[i]));
  }

  uint64_t opCount = 0;
  for (int op = begin_op; op < end_op; op ++) {
    const char *OpsName[] = {"Copy", "Local Copy", "Double Copy", "Reduce", "ReduceCopy"};
    printf("Testing %s: \n", OpsName[op]);
    // 2 warm up cycles
    for (int i = 0; i < 2; i ++) {
      for (int i = 0; i < nGpu; i ++) {
        HIPCHECK(hipSetDevice(i));
        //launch the kernel
        hipLaunchKernelGGL(flagSyncKerns[op*2 + sync],
            /*grid dim x,y,z*/        dim3(workgroups, 1, 1),
            /*block dim x,y,z*/       dim3(THREADS, 1, 1),
            /*dynamic shared mem*/    0,
            /*stream*/                stream[i],
            /*kernel args*/           transfer_data[i], d_profiling_data[i], opCount);
      }
      opCount++;
    }

    for (int i = 0; i < nGpu; i ++) {
      HIPCHECK(hipSetDevice(i));
      HIPCHECK(hipStreamSynchronize(stream[i]));
      HIPCHECK(hipMemset(d_profiling_data[i], 0, sizeof(struct profiling_data_t)));
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i ++) {
      for (int i = 0; i < nGpu; i ++) {
        HIPCHECK(hipSetDevice(i));
        //launch the kernel
        hipLaunchKernelGGL(flagSyncKerns[op*2 + sync],
            /*grid dim x,y,z*/        dim3(workgroups, 1, 1),
            /*block dim x,y,z*/       dim3(THREADS, 1, 1),
            /*dynamic shared mem*/    0,
            /*stream*/                stream[i],
            /*kernel args*/           transfer_data[i], d_profiling_data[i], opCount);
      }
      opCount++;
    }

    for (int i = 0; i < nGpu; i ++) {
      HIPCHECK(hipSetDevice(i));
      HIPCHECK(hipStreamSynchronize(stream[i]));
    }

    auto delta = std::chrono::high_resolution_clock::now() - start;
    double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();

    for (int i = 0; i < nGpu; i ++) {
      HIPCHECK(hipMemcpyAsync(profiling_data[i], d_profiling_data[i],
                              sizeof(struct profiling_data_t), hipMemcpyDeviceToHost,
                              stream[i]));
      HIPCHECK(hipStreamSynchronize(stream[i]));
#define RTC_CLOCK_FREQ 2.7E07
      double t0 = (double)profiling_data[i]->write_cycles/((double)RTC_CLOCK_FREQ)/(double)workgroups;
      fprintf(stderr, "GPU %d: time %.4fs bytes_transferred %lu kernel throughput %.2f GB/s\n",
        i, t0, profiling_data[i]->bytes_transferred, (double)profiling_data[i]->bytes_transferred/(t0*1.0E9));
    }

    double speed = (double)(profiling_data[0]->bytes_transferred) / (deltaSec*1.0E9);
    printf("Transfered %lu bytes in %f s. Throughput %f GB/s\n", profiling_data[0]->bytes_transferred, deltaSec, speed);
  }

  for (int i = 0; i < nGpu; i ++) {
    HIPCHECK(hipStreamDestroy(stream[i]));
    HIPCHECK(hipFree((void*) transfer_data[i]));
    for (int j = 0; j < workgroups; j++) {
      HIPCHECK(hipFree((void*) buff[i*MAX_WORKGROUPS+j]));
      HIPCHECK(hipFree((void*) buff_coarse[i*MAX_WORKGROUPS+j]));
    }
    HIPCHECK(hipFree((void*) d_profiling_data[i]));
    free(profiling_data[i]);
  }

  printf("opCount: ");
  for (int i = 0; i < nGpu; i++)
    printf("%ld ", remOpCount[i]);
  printf("\n");
  HIPCHECK(hipHostFree((void*)remOpCount));
}
