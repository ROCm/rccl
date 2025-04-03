typedef void(*ncclDevFuncPtr_t)();

__device__ ncclDevFuncPtr_t const ncclDevFuncTable_1[] = {
nullptr};
__device__ ncclDevFuncPtr_t const ncclDevFuncTable_2[] = {
nullptr};
__device__ ncclDevFuncPtr_t const ncclDevFuncTable_3[] = {
nullptr};
__device__ ncclDevFuncPtr_t const ncclDevFuncTable_4[] = {
nullptr};

template<unsigned short f, unsigned short l>
struct Caller1 {
  static __forceinline__ __device__ __host__
  void call1(unsigned short funcIndex) noexcept
  {
    constexpr unsigned short m = f + (l - f) / 2;
    return (funcIndex < m) ? Caller1<f, m>::call1(funcIndex) : Caller1<m, l>::call1(funcIndex);
  }
};

template<unsigned short f>
struct Caller1<f, f + 1>{
  static __forceinline__ __device__ __host__
  void call1(unsigned short funcIndex) noexcept { ncclDevFuncTable_1[f](); }
};
__forceinline__ __device__ void NCCL_CALL_FUNCTIONS_1(unsigned short funcIndex) noexcept {
  Caller1<0, 0>::call1(funcIndex);
}

template<unsigned short f, unsigned short l>
struct Caller2 {
  static __forceinline__ __device__ __host__
  void call2(unsigned short funcIndex) noexcept
  {
    constexpr unsigned short m = f + (l - f) / 2;
    return (funcIndex < m) ? Caller2<f, m>::call2(funcIndex) : Caller2<m, l>::call2(funcIndex);
  }
};

template<unsigned short f>
struct Caller2<f, f + 1>{
  static __forceinline__ __device__ __host__
  void call2(unsigned short funcIndex) noexcept { ncclDevFuncTable_2[f](); }
};
__forceinline__ __device__ void NCCL_CALL_FUNCTIONS_2(unsigned short funcIndex) noexcept {
  Caller2<0, 512>::call2(funcIndex);
}

template<unsigned short f, unsigned short l>
struct Caller4 {
  static __forceinline__ __device__ __host__
  void call4(unsigned short funcIndex) noexcept
  {
    constexpr unsigned short m = f + (l - f) / 2;
    return (funcIndex < m) ? Caller4<f, m>::call4(funcIndex) : Caller4<m, l>::call4(funcIndex);
  }
};

template<unsigned short f>
struct Caller4<f, f + 1>{
  static __forceinline__ __device__ __host__
  void call4(unsigned short funcIndex) noexcept { ncclDevFuncTable_4[f](); }
};
__forceinline__ __device__ void NCCL_CALL_FUNCTIONS_4(unsigned short funcIndex) noexcept {
  Caller4<0, 0>::call4(funcIndex);
}

