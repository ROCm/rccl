/*************************************************************************
 * Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_CHECKSUM_H_
#define NET_IB_CHECKSUM_H_

#include <stddef.h>
#include <stdint.h>

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

static inline uint32_t ncclIbQuickXorCsumHost(const void* data, size_t nbytes) {
  const uint8_t* bytes = (const uint8_t*)data;
  uint32_t csum = 0;
  size_t i = 0;
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
