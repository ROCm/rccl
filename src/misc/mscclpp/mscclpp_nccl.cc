/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt and NOTICES.txt for license information
 ************************************************************************/

#include "mscclpp/mscclpp_nccl.h"
#include "debug.h"
#include <dlfcn.h>
#include <unordered_map>

#define MSCCLPP_DECLARE(X) decltype(mscclpp_##X) mscclpp_##X = nullptr
#define MSCCLPP_LOAD(HANDLE, X) do {                            \
  (mscclpp_##X) = (decltype(mscclpp_##X))dlsym((HANDLE), (#X)); \
  const char* error;                                            \
  if ((error = dlerror()) != nullptr) {                         \
    WARN("MSCCL++: failed to load %s : %s", (#X), error);       \
    return false;                                               \
  }                                                             \
} while (false)

static const char mscclpp_nccl_lib_name[] = "libmscclpp_nccl.so";

MSCCLPP_DECLARE(ncclGetUniqueId);
MSCCLPP_DECLARE(ncclCommInitRank);
MSCCLPP_DECLARE(ncclCommDestroy);
MSCCLPP_DECLARE(ncclAllReduce);
MSCCLPP_DECLARE(ncclAllGather);

bool mscclpp_init() {
  void* handle = dlopen(mscclpp_nccl_lib_name, RTLD_LAZY);
  if (!handle) {
    WARN("MSCCL++: failed to access %s : %s", mscclpp_nccl_lib_name, dlerror());
    return false;
  }
  dlerror(); // Clear any errors.

  MSCCLPP_LOAD(handle, ncclGetUniqueId);
  MSCCLPP_LOAD(handle, ncclCommInitRank);
  MSCCLPP_LOAD(handle, ncclCommDestroy);
  MSCCLPP_LOAD(handle, ncclAllReduce);
  MSCCLPP_LOAD(handle, ncclAllGather);
  return true;
}

std::unordered_map<ncclUniqueId, mscclppUniqueId> mscclpp_uniqueIdMap;
