#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <string>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <iostream> //cerr
#include <cstring>
#include <emmintrin.h>  // For __m128i
#include <smmintrin.h>  // For _mm_stream_load_si128

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

#define MAX_BUFF_SIZE 67108864

void memcpy_sse_movntdqa(void *dst, const void *src, size_t len) {
  assert(((uintptr_t)dst&15) == 0 && ((uintptr_t)src&15) == 0 && (len%32) == 0);
  while (len >= 32) {
    __m128i *S = (__m128i *)src;
    __m128i *D = (__m128i *)dst;
    __m128i tmp[2];

    tmp[0] = _mm_stream_load_si128(S + 0);
    tmp[1] = _mm_stream_load_si128(S + 1);
    _mm_store_si128(D + 0, tmp[0]);
    _mm_store_si128(D + 1, tmp[1]);

    src = (uint8_t *)src + 32;
    dst = (uint8_t *)dst + 32;
    len -= 32;
  }
}

int main(int argc, char** argv) {
  void *gpu_buffer, *host_buffer;

  HIPCHECK(hipSetDevice(0));
  HIPCHECK(hipExtMallocWithFlags(&gpu_buffer, MAX_BUFF_SIZE, hipDeviceMallocUncached));
  HIPCHECK(hipHostMalloc(&host_buffer, MAX_BUFF_SIZE, hipHostAllocMapped));

  //warm up
  memcpy(gpu_buffer, host_buffer, MAX_BUFF_SIZE);
  memcpy(host_buffer, gpu_buffer, MAX_BUFF_SIZE);

  for (uint64_t size = 32; size <= MAX_BUFF_SIZE; size *= 2) {
    auto start = std::chrono::high_resolution_clock::now();
    memcpy(gpu_buffer, host_buffer, size);
    auto delta_1 = std::chrono::high_resolution_clock::now() - start;
    start = std::chrono::high_resolution_clock::now();
    memcpy(host_buffer, gpu_buffer, size);
    auto delta_2 = std::chrono::high_resolution_clock::now() - start;
    start = std::chrono::high_resolution_clock::now();
    memcpy_sse_movntdqa(host_buffer, gpu_buffer, size);
    auto delta_3 = std::chrono::high_resolution_clock::now() - start;
    auto bw_h2g = size/(std::chrono::duration_cast<std::chrono::duration<double>>(delta_1).count()*1E6);
    auto bw_g2h = size/(std::chrono::duration_cast<std::chrono::duration<double>>(delta_2).count()*1E6);
    auto bw_g2h_sse = size/(std::chrono::duration_cast<std::chrono::duration<double>>(delta_3).count()*1E6);
    printf("size %ld, CPU->GPU %.2f MB/s, GPU->CPU %.2f MB/s VMOVNTDQA %.2f MB/s\n", size, bw_h2g, bw_g2h, bw_g2h_sse);
  }

  HIPCHECK(hipFree(gpu_buffer));
  HIPCHECK(hipHostFree(host_buffer));
  return 0;
}
