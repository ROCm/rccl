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
    exit(1);                                                    \
  }                                                             \
} while (false)

static const char mscclpp_nccl_lib_name[] = "libmscclpp_nccl.so";

MSCCLPP_DECLARE(ncclGetUniqueId);
MSCCLPP_DECLARE(ncclCommInitRank);
MSCCLPP_DECLARE(ncclCommDestroy);
MSCCLPP_DECLARE(ncclAllReduce);
MSCCLPP_DECLARE(ncclAllGather);

static struct mscclpp_nccl_access {
  void* handle;
  
  mscclpp_nccl_access() : handle(nullptr) {
    Dl_info pathInfo;
    dladdr((void*)ncclCommInitRank, &pathInfo);
    INFO(NCCL_INIT, "***** %s *****", pathInfo.dli_fname);
    
    handle = dlopen(mscclpp_nccl_lib_name, RTLD_LAZY);
    if (!handle) {
      WARN("MSCCL++: failed to access %s : %s", mscclpp_nccl_lib_name, dlerror());
    }
    dlerror(); // Clear any errors.

    MSCCLPP_LOAD(handle, ncclGetUniqueId);
    MSCCLPP_LOAD(handle, ncclCommInitRank);
    MSCCLPP_LOAD(handle, ncclCommDestroy);
    MSCCLPP_LOAD(handle, ncclAllReduce);
    MSCCLPP_LOAD(handle, ncclAllGather);
  }
  ~mscclpp_nccl_access() {
    if (handle) {
      INFO(NCCL_INIT, "MSCCL++: closing handle to %s", mscclpp_nccl_lib_name);
      dlclose(handle);
    }
  }
} access;

std::unordered_map<ncclUniqueId, mscclpp_ncclUniqueId> mscclpp_uniqueIdMap;
