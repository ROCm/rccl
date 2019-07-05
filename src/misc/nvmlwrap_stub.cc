/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nvmlwrap.h"

ncclResult_t wrapNvmlSymbols(void) {
  return ncclSuccess;
}

ncclResult_t wrapNvmlInit(void) {
  return ncclSuccess;
}

ncclResult_t wrapNvmlShutdown(void) {
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index) {
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t* pci) {
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber) {
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) {
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
    nvmlNvLinkCapability_t capability, unsigned int *capResult) {
  return ncclSuccess;
}
