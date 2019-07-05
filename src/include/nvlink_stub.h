/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NVLINK_H_
#define NCCL_NVLINK_H_

#include <sys/stat.h>
#include <fcntl.h>
#include "nvmlwrap.h"
#include "topo.h"

#define CONNECT_NVLINK 0x10
#define CONNECT_NVSWITCH 0x100

enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceGpu,
  ncclNvLinkDeviceSwitch,
  ncclNvLinkDeviceBridge, // IBM/Power NVLink bridge (Device 04ea)
};

static int getNvlinkGpu(const char* busId1, const char* busId2) {
  int links = 0;
  return CONNECT_NVLINK*links;
}

#endif
