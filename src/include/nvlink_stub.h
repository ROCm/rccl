/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NVLINK_H_
#define NCCL_NVLINK_H_

#include "topo.h"

#define CONNECT_NVLINK 0x10
#define CONNECT_NVSWITCH 0x100

static int getNumNvlinks(const char* busId) {
  return 0;
}

#endif
