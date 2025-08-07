#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <string>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <iostream> //cerr
#include <cstring>

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

#define MAX_BUFF_SIZE 268435456

int main(int argc, char** argv) {
  void *gpu_buffer, *host_buffer;

  HIPCHECK(hipSetDevice(0));
  HIPCHECK(hipExtMallocWithFlags(&gpu_buffer, MAX_BUFF_SIZE, hipDeviceMallocUncached));
  HIPCHECK(hipHostMalloc(&host_buffer, MAX_BUFF_SIZE, hipHostAllocMapped));

  for (uint64_t size = 8; size <= MAX_BUFF_SIZE; size *= 2) {
    auto start = std::chrono::high_resolution_clock::now();
    memcpy(gpu_buffer, host_buffer, size);
    auto delta_1 = std::chrono::high_resolution_clock::now() - start;
    start = std::chrono::high_resolution_clock::now();
    memcpy(host_buffer, gpu_buffer, size);
    auto delta_2 = std::chrono::high_resolution_clock::now() - start;
    auto bw_h2g = size/(std::chrono::duration_cast<std::chrono::duration<double>>(delta_1).count()*1E6);
    auto bw_g2h = size/(std::chrono::duration_cast<std::chrono::duration<double>>(delta_2).count()*1E6);
    printf("size %ld, CPU->GPU %.2f MB/s, GPU->CPU %.2f MB/s\n", size, bw_h2g, bw_g2h);
  }

  HIPCHECK(hipFree(gpu_buffer));
  HIPCHECK(hipHostFree(host_buffer));
  return 0;
}
