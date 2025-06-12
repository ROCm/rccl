#ifndef DEVICE_TABLE_COMPATIBILITY
#define DEVICE_TABLE_COMPATIBILITY

struct rcclKernelItem {
  void* funcPtr;
  int   unroll;
};
static struct rcclKernelItem rcclKernelTable[] = { };

template <int unroll>
__forceinline__ __device__ void NCCL_CALL_FUNCTIONS(unsigned short funcIndex) noexcept { }

#endif
