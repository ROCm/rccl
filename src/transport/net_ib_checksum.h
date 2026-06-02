/*************************************************************************
 * Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_CHECKSUM_H_
#define NET_IB_CHECKSUM_H_

#include <stddef.h>
#include <stdint.h>

// x86 streaming-load (MOVNTDQA) intrinsics for the host-side XOR over the
// uncached net buffers. Host-only: the __x86_64__ guard keeps the device
// compilation pass (and non-x86 hosts) on the scalar fallback below.
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// Mirror the RCCL_IB_CHECKSUM_DEVICE_ENABLED gate from device.h. Both
// places use the same #ifndef pattern so a CMake -D override hits them
// identically (this header is included from host TUs that don't pull in
// device.h, so the gate must be definable here too).
#ifndef RCCL_IB_CHECKSUM_DEVICE_ENABLED
#define RCCL_IB_CHECKSUM_DEVICE_ENABLED 1
#endif

#if RCCL_IB_CHECKSUM_DEVICE_ENABLED

// RDMA WRITE WITH_IMM layout: size rarely needs the full 32 bits per step.
#define NCCL_IB_IMM_SIZE_BITS 20
#define NCCL_IB_IMM_CSUM_BITS 12
#define NCCL_IB_IMM_SIZE_MAX ((1u << NCCL_IB_IMM_SIZE_BITS) - 1)
#define NCCL_IB_IMM_CSUM_MASK ((1u << NCCL_IB_IMM_CSUM_BITS) - 1)
// Reserved IMM checksum field: checksum disabled (LL/LL128).
#define NCCL_IB_IMM_CSUM_DISABLED NCCL_IB_IMM_CSUM_MASK
// Proxy -> IB: do not pack or verify checksum for this slot.
#define NCCL_IB_CHECKSUM_NONE 0xffffffffu

static inline uint32_t ncclIbImmFoldCsum(uint32_t csum) {
  csum ^= csum >> 16;
  csum ^= csum >> NCCL_IB_IMM_CSUM_BITS;
  return csum & NCCL_IB_IMM_CSUM_MASK;
}

// XOR-fold of the first `nbytes` of `data`, matching the kernel's 32-bit
// word XOR (ncclQuickXorCsumWarp) so the send-side post-completion verify can
// compare against r->send.checksum. The net send/recv buffers on gfx94x/gfx950
// live in uncached (write-combining) host/device memory (hipHostMallocUncached
// / hipDeviceMallocUncached), where a plain scalar load issues one slow
// uncached transaction per word on the proxy critical path. On x86 we instead
// use MOVNTDQA (_mm_stream_load_si128): a streaming load that aggregates WC
// reads into the CPU's fill buffers, with four independent accumulators to
// keep several buffers in flight. XOR is associative+commutative, so the
// per-lane reduction of the 128-bit accumulators equals the sequential word
// XOR regardless of grouping.
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("sse4.1")))
#endif
static inline uint32_t ncclIbQuickXorCsumHost(const void* data, size_t nbytes) {
  const uint8_t* bytes = (const uint8_t*)data;
  uint32_t csum = 0;
  size_t i = 0;
#if defined(__x86_64__) || defined(__i386__)
  uintptr_t addr = (uintptr_t)bytes;
  // Only take the streaming path when the buffer is at least 4-byte aligned
  // (so the 16-byte-alignment head stays on word boundaries) and large enough
  // to amortize setup; otherwise fall through to the scalar loop for the whole
  // buffer.
  if ((addr & 3) == 0 && nbytes >= 64) {
    // Scalar 4-byte words until the pointer is 16-byte aligned. With a 4-byte
    // aligned base this head is exactly 0/4/8/12 bytes, so word boundaries
    // (and thus the XOR result) are preserved.
    size_t head = (size_t)((16 - (addr & 15)) & 15);
    for (; i < head; i += 4) {
      uint32_t w;
      __builtin_memcpy(&w, bytes + i, sizeof(w));
      csum ^= w;
    }
    __m128i a0 = _mm_setzero_si128(), a1 = _mm_setzero_si128();
    __m128i a2 = _mm_setzero_si128(), a3 = _mm_setzero_si128();
    for (; i + 64 <= nbytes; i += 64) {
      a0 = _mm_xor_si128(a0, _mm_stream_load_si128((__m128i*)(bytes + i +  0)));
      a1 = _mm_xor_si128(a1, _mm_stream_load_si128((__m128i*)(bytes + i + 16)));
      a2 = _mm_xor_si128(a2, _mm_stream_load_si128((__m128i*)(bytes + i + 32)));
      a3 = _mm_xor_si128(a3, _mm_stream_load_si128((__m128i*)(bytes + i + 48)));
    }
    for (; i + 16 <= nbytes; i += 16) {
      a0 = _mm_xor_si128(a0, _mm_stream_load_si128((__m128i*)(bytes + i)));
    }
    __m128i acc = _mm_xor_si128(_mm_xor_si128(a0, a1), _mm_xor_si128(a2, a3));
    uint32_t lanes[4];
    _mm_storeu_si128((__m128i*)lanes, acc);
    csum ^= lanes[0] ^ lanes[1] ^ lanes[2] ^ lanes[3];
    // Streaming loads are weakly ordered; drain the fill buffers before the
    // next completion reads (a possibly re-DMA'd) buffer through this path.
    _mm_mfence();
  }
#endif
  for (; i + 4 <= nbytes; i += 4) {
    uint32_t w;
    __builtin_memcpy(&w, bytes + i, sizeof(w));
    csum ^= w;
  }
  if (i < nbytes) {
    uint32_t tail = 0;
    size_t rem = nbytes - i;
    __builtin_memcpy(&tail, bytes + i, rem);
    csum ^= tail;
  }
  return csum;
}

static inline uint32_t ncclIbImmPack(int size, uint32_t csum) {
  if (csum == NCCL_IB_IMM_CSUM_DISABLED)
    return ((uint32_t)size & NCCL_IB_IMM_SIZE_MAX) | (NCCL_IB_IMM_CSUM_DISABLED << NCCL_IB_IMM_SIZE_BITS);
  uint32_t folded = ncclIbImmFoldCsum(csum);
  return ((uint32_t)size & NCCL_IB_IMM_SIZE_MAX) | (folded << NCCL_IB_IMM_SIZE_BITS);
}

static inline int ncclIbImmUnpackSize(uint32_t imm) {
  return (int)(imm & NCCL_IB_IMM_SIZE_MAX);
}

static inline uint32_t ncclIbImmUnpackCsum(uint32_t imm) {
  return (imm >> NCCL_IB_IMM_SIZE_BITS) & NCCL_IB_IMM_CSUM_MASK;
}

static inline int ncclIbImmCanPack(int size) {
  return size >= 0 && (uint32_t)size <= NCCL_IB_IMM_SIZE_MAX;
}

// Called from net proxy before isend; slot matches IB fifo slot (step % MAX_REQUESTS).
void ncclIbSetProxyChecksum(void* sendComm, int slot, uint32_t checksum);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
void rocmIbSetProxyChecksum(void* sendComm, int slot, uint32_t checksum);
#endif

// Called from net proxy after recv test() returns done; copies the wire-
// received 12-bit folded checksums (or NCCL_IB_CHECKSUM_NONE when no IMM
// checksum was attached) for up to n multi-recv subs into checksums[].
// Must be called before any subsequent test() on the same recvComm so the
// per-comm stash is not overwritten by the next completion.
ncclResult_t ncclIbGetRecvChecksums(void* recvComm, int n, uint32_t* checksums);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
ncclResult_t rocmIbGetRecvChecksums(void* recvComm, int n, uint32_t* checksums);
#endif

#endif // RCCL_IB_CHECKSUM_DEVICE_ENABLED

#endif // NET_IB_CHECKSUM_H_
