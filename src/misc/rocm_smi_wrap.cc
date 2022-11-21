/*
Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include "rocm_smi_wrap.h"
#include "core.h"
#include "utils.h"

#define ROCMSMICHECK(cmd) do {               \
  rsmi_status_t ret = cmd;                   \
  if( ret != RSMI_STATUS_SUCCESS ) {         \
    const char *err;                         \
    rsmi_status_string(ret, &err);           \
    WARN("ROCm SMI init failure %s", err);   \
    return ncclInternalError;                \
  }                                          \
} while(false)


ncclResult_t rocm_smi_init() {
  ROCMSMICHECK(rsmi_init(0));
  rsmi_version_t version;
  ROCMSMICHECK(rsmi_version_get(&version));
  INFO(NCCL_INIT, "rocm_smi_lib: version %d.%d.%d.%s", version.major, version.minor, version.patch, version.build);
  return ncclSuccess;
}

ncclResult_t rocm_smi_getNumDevice(uint32_t* num_devs) {
  ROCMSMICHECK(rsmi_num_monitor_devices(num_devs));
  return ncclSuccess;
}

ncclResult_t rocm_smi_getDevicePciBusIdString(uint32_t deviceIndex, char* busId, size_t len) {
  uint64_t id;
  ROCMSMICHECK(rsmi_dev_pci_id_get(deviceIndex, &id));
  /** rocm_smi's bus ID format
   *  | Name     | Field   |
   *  ---------- | ------- |
   *  | Domain   | [64:32] |
   *  | Reserved | [31:16] |
   *  | Bus      | [15: 8] |
   *  | Device   | [ 7: 3] |
   *  | Function | [ 2: 0] |
   **/
  snprintf(busId, len, "%04lx:%02lx:%02lx.%01lx", (id) >> 32, (id & 0xff00) >> 8, (id & 0xf8) >> 3, (id & 0x7));
  return ncclSuccess;
}


ncclResult_t rocm_smi_getDeviceIndexByPciBusId(const char* pciBusId, uint32_t* deviceIndex) {
  uint32_t i, num_devs = 0;
  int64_t busid;

  busIdToInt64(pciBusId, &busid);
  /** convert to rocm_smi's bus ID format
   *  | Name     | Field   |
   *  ---------- | ------- |
   *  | Domain   | [64:32] |
   *  | Reserved | [31:16] |
   *  | Bus      | [15: 8] |
   *  | Device   | [ 7: 3] |
   *  | Function | [ 2: 0] |
   **/
  busid = ((busid&0xffff00000L)<<12)+((busid&0xff000L)>>4)+((busid&0xff0L)>>1)+(busid&0x7L);
  ROCMSMICHECK(rsmi_num_monitor_devices(&num_devs));
  for (i = 0; i < num_devs; i++) {
    uint64_t bdfid;
    ROCMSMICHECK(rsmi_dev_pci_id_get(i, &bdfid));
    if (bdfid == busid) break;
  }

  if (i < num_devs) {
    *deviceIndex = i;
    return ncclSuccess;
  }
  else {
    WARN("rocm_smi_lib: %s device index not found", pciBusId);
    return ncclInternalError;
  }
}

ncclResult_t rocm_smi_getLinkInfo(int srcIndex, int dstIndex, RSMI_IO_LINK_TYPE* rsmi_type, int *hops, int *count) {
  uint64_t rsmi_hops, rsmi_weight;
  ROCMSMICHECK(rsmi_topo_get_link_type(srcIndex, dstIndex, &rsmi_hops, rsmi_type));
  ROCMSMICHECK(rsmi_topo_get_link_weight(srcIndex, dstIndex, &rsmi_weight));
  *hops = 2;
  *count = 1;
  if (*rsmi_type == RSMI_IOLINK_TYPE_XGMI && rsmi_weight == 15) {
    *hops = 1;
#if defined USE_ROCM_SMI64CONFIG && rocm_smi_VERSION_MAJOR >= 5
    uint64_t min_bw = 0, max_bw = 0;
    rsmi_version_t version;
    ROCMSMICHECK(rsmi_version_get(&version));
    if (version.major >= 5)
      ROCMSMICHECK(rsmi_minmax_bandwidth_get(srcIndex, dstIndex, &min_bw, &max_bw));
    if (max_bw && min_bw)
      *count = max_bw/min_bw;
#endif
  }
  return ncclSuccess;
}
