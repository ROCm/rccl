/*************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NVTX_STUB_H_
#define NCCL_NVTX_STUB_H_

struct nccl_domain{static constexpr char const* name{"NCCL"};};

#define NVTX3_FUNC_RANGE_IN(domain)
#define nvtxNameOsThreadA(syscall, thread)

#endif
