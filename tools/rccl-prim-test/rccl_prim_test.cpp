/*
Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

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
#define MAX_WORKGROUPS 32
#define THREADS 256

#define COPY_UNROLL       4
#define REDUCE_UNROLL     2
#define DOUBLECOPY_UNROLL 2
#define REDUCECOPY_UNROLL 2
#define ALL2ALL_UNROLL    2



#define RST  "\x1B[0m"
#define KBLU  "\x1B[34m"
#define FBLU(x) KBLU x RST
#define BOLD(x) "\x1B[1m" x RST

#define RTC_CLOCK_FREQ_VEGA20 2.5E07
//Right now kept the MI100 RTC frequency same as Vega20
//as we are not aware of MI100 frequency, once we we come to know about it
//we will update it.
#define RTC_CLOCK_FREQ_MI100 2.5E07
#define RTC_CLOCK_FREQ_DEFAULT 2.7E07

__device__
inline  __attribute((always_inline))
long long int __rtc64() {
#if __HIP__
  return (long long int) __builtin_amdgcn_s_memrealtime();
#else
  return (long long int) __clock_u64();
#endif
}

struct transfer_data_t {
  // Buffers for all OPs except all to all
  float *dest0[MAX_WORKGROUPS]; //remote fine grain
  float *src0[MAX_WORKGROUPS];  //local fine grain
  float *dest1[MAX_WORKGROUPS]; //local coarse grain
  float *src1[MAX_WORKGROUPS];  //local coarse grain
  // Buffers for all to all
  const float *srcs[MAX_WORKGROUPS][MAX_GPU];
  float *dsts[MAX_WORKGROUPS][MAX_GPU];
  int N;
  int gpu;
  int ngpu;
};

struct profiling_data_t {
  uint64_t write_cycles[MAX_WORKGROUPS];
  uint64_t bytes_transferred[MAX_WORKGROUPS];
};


#define LOAD(VAR) __atomic_load_n((VAR), __ATOMIC_SEQ_CST)
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)

void print_table_header(void) {
  fprintf(stderr, "%120s","=================================================================================================================================\n");
  fprintf(stderr, "%-20s %-13s %-13s %-13s %-13s %-20s %-20s\n","[Originating GPU]", "[Directions]", "[WorkGroup]", "[linktype]", "[time(sec)]" , "[bytes_transferred]",  "[kernel throughput(GB/s)]");
  fprintf(stderr, "%120s","=================================================================================================================================\n");
}

void print_table_summary_line(void) {
  fprintf(stderr, "%120s","---------------------------------------------------------------------------------------------------------------------------------\n");
}

enum Ops {
  OP_COPY,
  OP_LOCALCOPY,
  OP_DOUBLECOPY,
  OP_REDUCE,
  OP_REDUCECOPY,
  OP_READ,
  OP_ALL2ALL,
  NUM_OPS,
};

template<int op, int NGPUS>
__global__ void flag_sync_kernel(struct transfer_data_t* transfer_data, struct profiling_data_t* profiling_data, uint64_t opCount) {
  size_t tid = threadIdx.x;
  uint64_t curr_time, next_time;
  int bid = blockIdx.x;
  int n = transfer_data->N;

  const float *srcs[NGPUS];
  float *dsts[NGPUS];

  if (tid == 0) {
    curr_time = __rtc64();
  }

  if (op == OP_COPY) {
    srcs[0] = transfer_data->src0[bid];
    dsts[0] = transfer_data->dest0[bid];
    ReduceOrCopyMulti<COPY_UNROLL, FuncPassA<float>, float, 1, 1, 1, 1>(threadIdx.x, THREADS,
      1, srcs, 1, dsts, n);
  }
  if (op == OP_LOCALCOPY) {
    srcs[0] = transfer_data->src0[bid];
    dsts[0] = transfer_data->dest1[bid];
    ReduceOrCopyMulti<COPY_UNROLL, FuncPassA<float>, float, 1, 1, 1, 1>(threadIdx.x, THREADS,
      1, srcs, 1, dsts, n);
  }
  if (op == OP_DOUBLECOPY) {
    srcs[0] = transfer_data->src0[bid];
    dsts[0] = transfer_data->dest0[bid];
    dsts[1] = transfer_data->dest1[bid];
    ReduceOrCopyMulti<DOUBLECOPY_UNROLL, FuncPassA<float>, float, 1, 1, 1, 2>(threadIdx.x, THREADS,
      1, srcs, 2, dsts, n);
  }
  if (op == OP_REDUCE) {
    srcs[0] = transfer_data->src0[bid];
    srcs[1] = transfer_data->src1[bid];
    dsts[0] = transfer_data->dest0[bid];
    ReduceOrCopyMulti<REDUCE_UNROLL, FuncSum<float>, float, 1, 2, 1, 1>(threadIdx.x, THREADS,
      2, srcs, 1, dsts, n);
  }
  if (op == OP_REDUCECOPY) {
    srcs[0] = transfer_data->src0[bid];
    srcs[1] = transfer_data->src1[bid];
    dsts[0] = transfer_data->dest0[bid];
    dsts[1] = transfer_data->dest1[bid];
    ReduceOrCopyMulti<REDUCECOPY_UNROLL, FuncSum<float>, float, 1, 2, 1, 2>(threadIdx.x, THREADS,
      2, srcs, 2, dsts, n);
  }
  if (op == OP_READ) {
    // Swapped the dest0 and src0 in passed parameter of copy kernel so that it can utilized for as a read kernel.
    // fetch op will happen on transfer_data->dest0[bid] and store op will happen on transfer_data->src0[bid]
    srcs[0] = transfer_data->dest0[bid];
    dsts[0] = transfer_data->src0[bid];
    ReduceOrCopyMulti<COPY_UNROLL, FuncPassA<float>, float, 1, 1, 1, 1>(threadIdx.x, THREADS,
      1, srcs, 1, dsts, n);
  }
  if (op == OP_ALL2ALL) {
    for (int i = 0; i < NGPUS; i++) {
      srcs[i] = transfer_data->srcs[bid][i];
      dsts[i] = transfer_data->dsts[bid][i];
    }
    ReduceOrCopyMulti<ALL2ALL_UNROLL, FuncSum<float>, float, 1, NGPUS, 1, NGPUS>(tid, THREADS,
      NGPUS, srcs, NGPUS, dsts, n);
  }

  __syncthreads();
  if (tid == 0) {
    next_time = __rtc64();
    __atomic_fetch_add(&(profiling_data->write_cycles[bid]), next_time - curr_time, __ATOMIC_SEQ_CST);
    // for all to all, read and write n itmes to all other GPUs, thus "n * sizeof(float) * (transfer_data->ngpu - 1) * 2" bytes
    if (op == OP_ALL2ALL) __atomic_fetch_add(&(profiling_data->bytes_transferred[bid]), n * sizeof(float) * (transfer_data->ngpu - 1) * 2, __ATOMIC_SEQ_CST);
    else __atomic_fetch_add(&(profiling_data->bytes_transferred[bid]), n * sizeof(float), __ATOMIC_SEQ_CST);
  }
}

typedef void(*flag_sync_kernel_t)(struct transfer_data_t* transfer_data, struct profiling_data_t* profiling_data, uint64_t opCount);

static flag_sync_kernel_t const flagSyncKerns[NUM_OPS+1] = {
  flag_sync_kernel<OP_COPY, 2>,
  flag_sync_kernel<OP_LOCALCOPY, 2>,
  flag_sync_kernel<OP_DOUBLECOPY, 2>,
  flag_sync_kernel<OP_REDUCE, 2>,
  flag_sync_kernel<OP_REDUCECOPY, 2>,
  flag_sync_kernel<OP_READ, 2>,
  flag_sync_kernel<OP_ALL2ALL, 4>,
  flag_sync_kernel<OP_ALL2ALL, 8>,
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

static void setupPeers(uint32_t *info, bool* is_xgmi, bool* is_2h4p) {
  int deviceCnt, dev;

  *is_xgmi = *is_2h4p = 0;
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
        if (*is_xgmi == 0 && linktype == 4) *is_xgmi = 1;
      }
      else
        info[i*deviceCnt+j] = 0;
    }
  }
  if (*is_xgmi && deviceCnt == 8) {
    uint32_t linktype, hop;
    HIPCHECK(hipExtGetLinkTypeAndHopCount(0, 4, &linktype, &hop));
    if (linktype != 4) *is_2h4p = 1;
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
  ring_1[0] =0;
  for (int i = 1; i < deviceCnt; i++)
    ring_1[i] = ring_0[deviceCnt-i];
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


static const char* link_type_name[] = {"HT", "QPI", "PCIE", "IB", "XGMI"};


int main(int argc,char* argv[])
{
  if (cmdOptionExists(argv, argv + argc, "-h")) {
    printf("./rccl_prim_test -w num_workgroups -p copy|localcopy|doublecopy|reduce|reducecopy|all2all -i iterations -n bytes -r \"0 1 2 3|3 2 1 0\"\n");
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

  char *r = getCmdOption(argv, argv + argc, "-r");
  if (r) printf("User specified ring topology: %s\n", r);

  const char *ops[] = {"copy", "localcopy", "doublecopy", "reduce", "reducecopy", "read", "all2all"};
  char *prim = getCmdOption(argv, argv + argc, "-p");
  int op = NUM_OPS, begin_op, end_op;
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

  int nGpu = 1;
  HIPCHECK(hipGetDeviceCount(&nGpu));

  uint32_t connection_info[MAX_GPU*MAX_GPU];
  // Enable peer access
  bool is_xgmi, is_2h4p;
  setupPeers(connection_info, &is_xgmi, &is_2h4p);
  hipDeviceProp_t prop;
  HIPCHECK(hipGetDeviceProperties(&prop, 0));
  static const char *ring_4p3l = "0 1 2 3|0 1 3 2|0 2 1 3|0 2 3 1|0 3 1 2|0 3 2 1";
  static const char *ring_8p6l = "0 4 5 6 7 3 2 1|0 7 4 3 5 1 6 2|0 6 4 2 5 7 1 3|0 1 2 3 7 6 5 4|0 2 6 1 5 3 4 7|0 3 1 7 5 2 4 6";
  if (prop.gcnArch == 908) {
    if (nGpu == 4 && is_xgmi) r = (char *)ring_4p3l;
    if (nGpu == 8 && is_xgmi && !is_2h4p) r = (char *)ring_8p6l;
  }

  // clockwise and counter clockwise rings
  int ring[MAX_WORKGROUPS][MAX_GPU];
  for (int i = 0; i < MAX_WORKGROUPS; i++)
    for (int j = 0; j <MAX_GPU; j++)
      ring[i][j] =  -1;

  int num_rings = 0;
  if (r) {
    int j = 0, n = 0;
    do {
      if (r[n] == ' ') continue;
      if (r[n] == '|') {
        num_rings ++;
        j = 0;
        continue;
      }
      ring[num_rings][j++] = r[n] - '0';
    } while (r[n++] != 0x0);
    num_rings ++;
  } else {
    setupRings(connection_info, ring[0], ring[1]);
    num_rings = 2;
  }

  // duplicate rings
  for (int i = num_rings; i < MAX_WORKGROUPS; i++) {
    for (int j = 0; j <MAX_GPU; j++)
      ring[i][j] =  ring[i%num_rings][j];
  }

  // data buffers
  float *buff[MAX_GPU*MAX_WORKGROUPS], *buff_coarse[MAX_GPU*MAX_WORKGROUPS];
  struct transfer_data_t h_transfer_data[MAX_GPU], *transfer_data[MAX_GPU];
  struct profiling_data_t *profiling_data[MAX_GPU], *d_profiling_data[MAX_GPU];
  hipStream_t stream[MAX_GPU];

  // print rings
  for (int i = 0; i < workgroups; i++) {
    printRing(i, ring[i], nGpu);
  }

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
      next_gpu = findNextGpu(ring[j], i, nGpu);
      //printf("GPU %d Ring %d -> Next GPU %d\n", i, j, next_gpu);
      h_transfer_data[i].dest0[j] = buff[next_gpu*MAX_WORKGROUPS+j] + N;
      h_transfer_data[i].dest1[j] = buff_coarse[i*MAX_WORKGROUPS+j] + N;
      h_transfer_data[i].src0[j] = buff[i*MAX_WORKGROUPS+j];
      h_transfer_data[i].src1[j] = buff_coarse[i*MAX_WORKGROUPS+j];
    }
    h_transfer_data[i].N = N;
    h_transfer_data[i].gpu = i;
    h_transfer_data[i].ngpu = nGpu;
  }

  for (int i = 0; i < nGpu; i ++) {
    for (int j = 0; j < workgroups; j++) {
      for (int k = 0; k < nGpu; k++) {
        h_transfer_data[i].srcs[j][k] = buff[((i+k)%nGpu)*MAX_WORKGROUPS+j];
        h_transfer_data[i].dsts[j][k] = buff[((i+k)%nGpu)*MAX_WORKGROUPS+j] + N;
        //printf("Setup GPU %d bid %d srcs[%d] %p dsts[%d] %p\n", i, j, k, h_transfer_data[i].srcs[j][k], k, h_transfer_data[i].dsts[j][k]);
      }
    }
  }

  for (int i = 0; i < nGpu; i ++) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMemcpyAsync(transfer_data[i], &h_transfer_data[i],
                            sizeof(struct transfer_data_t), hipMemcpyHostToDevice,
                            stream[i]));
    HIPCHECK(hipStreamSynchronize(stream[i]));
  }

  void *args[MAX_GPU*3];
  hipLaunchParams *launchParamsList= reinterpret_cast<hipLaunchParams *>(
            malloc(sizeof(hipLaunchParams)*MAX_GPU));

  uint64_t opCount = 0;
  for (int op = begin_op; op < end_op; op ++) {
    if (op == OP_ALL2ALL && nGpu != 4 && nGpu != 8) {
      printf("\n%s only supports 4 or 8 GPUs.\n", ops[op]);
      continue;
    }
    printf("\n[Testing %s]: \n", ops[op]);
    // 4 warm up cycles
    for (int j = 0; j < 4; j ++) {
      for (int i = 0; i < nGpu; i ++) {
        args[i*3] = &transfer_data[i];
        args[i*3+1] = &d_profiling_data[i];
        args[i*3+2] = &opCount;
        if (op == OP_ALL2ALL)
          launchParamsList[i].func = reinterpret_cast<void *>(flagSyncKerns[op + (nGpu/8)]);
        else
          launchParamsList[i].func = reinterpret_cast<void *>(flagSyncKerns[op]);
        launchParamsList[i].gridDim   = dim3(workgroups, 1, 1),
        launchParamsList[i].blockDim  = dim3(THREADS, 1, 1),
        launchParamsList[i].sharedMem = 0;
        launchParamsList[i].stream    = stream[i];
        launchParamsList[i].args      = args + i*3;
      }
      hipExtLaunchMultiKernelMultiDevice(launchParamsList, nGpu, 0);
      opCount++;
    }

    for (int i = 0; i < nGpu; i ++) {
      HIPCHECK(hipSetDevice(i));
      HIPCHECK(hipStreamSynchronize(stream[i]));
      HIPCHECK(hipMemset(d_profiling_data[i], 0, sizeof(struct profiling_data_t)));
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < iters; j ++) {
      for (int i = 0; i < nGpu; i ++) {
        args[i*3] = &transfer_data[i];
        args[i*3+1] = &d_profiling_data[i];
        args[i*3+2] = &opCount;
        if (op == OP_ALL2ALL)
          launchParamsList[i].func = reinterpret_cast<void *>(flagSyncKerns[op + (nGpu/8)]);
        else
          launchParamsList[i].func = reinterpret_cast<void *>(flagSyncKerns[op]);
        launchParamsList[i].gridDim   = dim3(workgroups, 1, 1),
        launchParamsList[i].blockDim  = dim3(THREADS, 1, 1),
        launchParamsList[i].sharedMem = 0;
        launchParamsList[i].stream    = stream[i];
        launchParamsList[i].args      = args + i*3;
      }
      hipExtLaunchMultiKernelMultiDevice(launchParamsList, nGpu, 0);
      opCount++;
    }

    for (int i = 0; i < nGpu; i ++) {
      HIPCHECK(hipSetDevice(i));
      HIPCHECK(hipStreamSynchronize(stream[i]));
    }

    auto delta = std::chrono::high_resolution_clock::now() - start;
    double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
    std::cout << BOLD(FBLU("[GPU to GPU Transfer Profiling Data]"))<<std::endl;
    print_table_header();
    for (int i = 0; i < nGpu; i ++) {
      HIPCHECK(hipMemcpyAsync(profiling_data[i], d_profiling_data[i],
                              sizeof(struct profiling_data_t), hipMemcpyDeviceToHost,
                              stream[i]));
      HIPCHECK(hipStreamSynchronize(stream[i]));

      uint64_t write_cycle = 0;
      uint64_t bytes_transferred = 0;

      hipDeviceProp_t prop;
      HIPCHECK(hipGetDeviceProperties(&prop, i));
      for (int j = 0; j < workgroups; j++) {
        int next_gpu;
        next_gpu = findNextGpu(ring[j], i, nGpu);

        uint32_t linktype;
        uint32_t hopcount;
        HIPCHECK(hipExtGetLinkTypeAndHopCount(i, next_gpu , &linktype, &hopcount));

        if (op == OP_ALL2ALL) {
          if(prop.gcnArch == 906) {
            write_cycle = write_cycle + profiling_data[i]->write_cycles[j];
            bytes_transferred = bytes_transferred + profiling_data[i]->bytes_transferred[j];
            double t0 = (double)profiling_data[i]->write_cycles[j]/((double)RTC_CLOCK_FREQ_VEGA20);
            fprintf(stderr, "%-20d %-d<->all       %-13d %-13s %-13.4f  %-20lu  %-.2f\n",
              i, i, j, link_type_name[linktype], t0, profiling_data[i]->bytes_transferred[j], (double)profiling_data[i]->bytes_transferred[j]/(t0*1.0E9));
          } else if (prop.gcnArch == 908) {
            write_cycle = write_cycle + profiling_data[i]->write_cycles[j];
            bytes_transferred = bytes_transferred + profiling_data[i]->bytes_transferred[j];
            double t0 = (double)profiling_data[i]->write_cycles[j]/((double)RTC_CLOCK_FREQ_MI100);
            fprintf(stderr, "%-20d %-d<->all       %-13d %-13s %-13.4f  %-20lu  %-.2f\n",
              i, i, j, link_type_name[linktype], t0, profiling_data[i]->bytes_transferred[j], (double)profiling_data[i]->bytes_transferred[j]/(t0*1.0E9));
          } else {
            write_cycle = write_cycle + profiling_data[i]->write_cycles[j];
            bytes_transferred = bytes_transferred + profiling_data[i]->bytes_transferred[j];
            double t0 = (double)profiling_data[i]->write_cycles[j]/((double)RTC_CLOCK_FREQ_DEFAULT);
            fprintf(stderr, "%-20d %-d<->all       %-13d %-13s %-13.4f  %-20lu  %-.2f\n",
              i, i, j, link_type_name[linktype], t0, profiling_data[i]->bytes_transferred[j], (double)profiling_data[i]->bytes_transferred[j]/(t0*1.0E9));
          }
        } else {
          if(prop.gcnArch == 906) {
            write_cycle = write_cycle + profiling_data[i]->write_cycles[j];
            bytes_transferred = bytes_transferred + profiling_data[i]->bytes_transferred[j];
            double t0 = (double)profiling_data[i]->write_cycles[j]/((double)RTC_CLOCK_FREQ_VEGA20);
            fprintf(stderr, "%-20d %-d->%-10d %-13d %-13s %-13.4f  %-20lu  %-.2f\n",
              i, i, next_gpu, j, link_type_name[linktype], t0, profiling_data[i]->bytes_transferred[j], (double)profiling_data[i]->bytes_transferred[j]/(t0*1.0E9));
          } else if (prop.gcnArch == 908) {
            write_cycle = write_cycle + profiling_data[i]->write_cycles[j];
            bytes_transferred = bytes_transferred + profiling_data[i]->bytes_transferred[j];
            double t0 = (double)profiling_data[i]->write_cycles[j]/((double)RTC_CLOCK_FREQ_MI100);
            fprintf(stderr, "%-20d %-d->%-10d %-13d %-13s %-13.4f  %-20lu  %-.2f\n",
              i, i, next_gpu, j, link_type_name[linktype], t0, profiling_data[i]->bytes_transferred[j], (double)profiling_data[i]->bytes_transferred[j]/(t0*1.0E9));
          } else {
            write_cycle = write_cycle + profiling_data[i]->write_cycles[j];
            bytes_transferred = bytes_transferred + profiling_data[i]->bytes_transferred[j];
            double t0 = (double)profiling_data[i]->write_cycles[j]/((double)RTC_CLOCK_FREQ_DEFAULT);
            fprintf(stderr, "%-20d %-d->%-10d %-13d %-13s %-13.4f  %-20lu  %-.2f\n",
              i, i, next_gpu, j, link_type_name[linktype], t0, profiling_data[i]->bytes_transferred[j], (double)profiling_data[i]->bytes_transferred[j]/(t0*1.0E9));
          }
        }
      }
      print_table_summary_line();
      double total = 0;
      if(prop.gcnArch == 906 ) {
        total = (double)write_cycle/((double)RTC_CLOCK_FREQ_VEGA20)/(double)workgroups;
      }else if (prop.gcnArch == 908 ){
        total = (double)write_cycle/((double)RTC_CLOCK_FREQ_MI100)/(double)workgroups;
      } else {
        total = (double)write_cycle/((double)RTC_CLOCK_FREQ_DEFAULT)/(double)workgroups;
      }
      fprintf(stderr, " %-61s %-13.4f  %-20lu  %-.2f\n",
        "Total" , total, bytes_transferred, (double)bytes_transferred/(total*1.0E9));
      print_table_summary_line();
    }
    std::cout << BOLD(FBLU("[Application Level Transfer Profiling Data]"))<<std::endl;
    uint64_t total_bytes_transferred = profiling_data[0]->bytes_transferred[0] * workgroups ;
    print_table_summary_line();
    fprintf(stderr, " %-61s %-13.4f  %-20lu  %-.2f\n",
      "Total" , deltaSec, total_bytes_transferred, (double)total_bytes_transferred/(deltaSec*1.0E9));
    print_table_summary_line();
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
}
