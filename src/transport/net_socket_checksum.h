/*************************************************************************
 * Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_SOCKET_CHECKSUM_H_
#define NET_SOCKET_CHECKSUM_H_

#include <stddef.h>
#include <stdint.h>

// Mirror the RCCL_IB_CHECKSUM_DEVICE_ENABLED gate from device.h /
// net_ib_checksum.h with the same #ifndef pattern so a CMake -D
// override hits all three identically.
#ifndef RCCL_IB_CHECKSUM_DEVICE_ENABLED
#define RCCL_IB_CHECKSUM_DEVICE_ENABLED 1
#endif

#if RCCL_IB_CHECKSUM_DEVICE_ENABLED

// Sentinel carried on the wire when no checksum is being conveyed
// (e.g. LL/LL128 proxy slots, or RCCL_SOCKET_CHECKSUM=0 on the sender).
// Numerically equal to NCCL_IB_CHECKSUM_NONE; the two transports use
// independent sentinels because they share a value but not a wire format.
#define NCCL_NET_SOCKET_CHECKSUM_NONE 0xffffffffu

// Match the device-side ncclQuickXorCsum: little-endian 4-byte XOR with a
// zero-padded tail. Used to recompute the checksum on the receive side and
// compare against the value carried in the per-step socket header.
static inline uint32_t ncclNetSocketQuickXorCsumHost(const void* data, size_t nbytes) {
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

// Called from the net proxy before each Isend on a socket sendComm. The
// checksum is consumed by the next ncclNetSocketIsend on this sendComm and
// the per-comm slot is reset to NCCL_NET_SOCKET_CHECKSUM_NONE so a stale
// value cannot leak into a subsequent send.
void ncclNetSocketSetProxyChecksum(void* sendComm, int slot, uint32_t checksum);

// Called from the net proxy after a recv test() returns done. Copies the
// most recently stashed wire-received checksum (already folded to 12 bits to
// match the IB IMM format consumed by the kernel; see
// ncclNetSocketStashRecvChecksum) into checksums[0]; remaining entries are
// padded with NCCL_NET_SOCKET_CHECKSUM_NONE because the socket plugin only
// supports single-buffer recvs. Must be called before any subsequent test()
// on the same recvComm so the per-comm stash is not overwritten.
ncclResult_t ncclNetSocketGetRecvChecksum(void* recvComm, int n, uint32_t* checksums);

#endif // RCCL_IB_CHECKSUM_DEVICE_ENABLED

#endif // NET_SOCKET_CHECKSUM_H_
