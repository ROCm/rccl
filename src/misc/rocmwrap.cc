/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "debug.h"
#include "rocmwrap.h"

#include <dlfcn.h>

#define DECLARE_ROCM_PFN(symbol) PFN_##symbol pfn_##symbol = nullptr

DECLARE_ROCM_PFN(hsa_amd_portable_export_dmabuf); // DMA-BUF support

/* ROCr Driver functions loaded with dlsym() */
DECLARE_ROCM_PFN(hsa_init);
DECLARE_ROCM_PFN(hsa_system_get_info);
DECLARE_ROCM_PFN(hsa_status_string);

static enum { hsaUninitialized, hsaInitializing, hsaInitialized, hsaError } hsaState = hsaUninitialized;

static void *hsaLib;
static uint16_t version_major, version_minor;
bool ncclCudaLaunchBlocking = false;
bool dmaBufSupport = false;

ncclResult_t rocmLibraryInit(void) {
  do {
    char* val = getenv("CUDA_LAUNCH_BLOCKING");
    ncclCudaLaunchBlocking = val!=nullptr && val[0]!=0 && !(val[0]=='0' && val[1]==0);
  } while (0);

  hsa_status_t res;

  if (hsaState == hsaInitialized)
    return ncclSuccess;
  if (hsaState == hsaError)
    return ncclSystemError;

  if (__sync_bool_compare_and_swap(&hsaState, hsaUninitialized, hsaInitializing) == false) {
    // Another thread raced in front of us. Wait for it to be done.
    while (hsaState == hsaInitializing) sched_yield();
    return (hsaState == hsaInitialized) ? ncclSuccess : ncclSystemError;
  }

  /*
   * Load ROCr driver library
   */
  char path[1024];
  char *ncclCudaPath = getenv("RCCL_ROCR_PATH");
  if (ncclCudaPath == NULL)
    snprintf(path, 1024, "%s", "libhsa-runtime64.so");
  else
    snprintf(path, 1024, "%s%s", ncclCudaPath, "libhsa-runtime64.so");

  hsaLib = dlopen(path, RTLD_LAZY);
  if (hsaLib == NULL) {
    WARN("Failed to find ROCm runtime library in %s (RCCL_ROCR_PATH=%s)", ncclCudaPath, ncclCudaPath);
    goto error;
  }

  /*
   * Load initial ROCr functions
   */

  pfn_hsa_init = (PFN_hsa_init) dlsym(hsaLib, "hsa_init");
  if (pfn_hsa_init == NULL) {
    WARN("Failed to load ROCr missing symbol hsa_init");
    goto error;
  }

  pfn_hsa_system_get_info = (PFN_hsa_system_get_info) dlsym(hsaLib, "hsa_system_get_info");
  if (pfn_hsa_system_get_info == NULL) {
    WARN("Failed to load ROCr missing symbol hsa_system_get_info");
    goto error;
  }

  pfn_hsa_status_string = (PFN_hsa_status_string) dlsym(hsaLib, "hsa_status_string");
  if (pfn_hsa_status_string == NULL) {
    WARN("Failed to load ROCr missing symbol hsa_status_string");
    goto error;
  }

  res = pfn_hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &version_major);
  if (res != 0) {
    WARN("pfn_hsa_system_get_info failed with %d", res);
    goto error;
  }
  res = pfn_hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &version_minor);
  if (res != 0) {
    WARN("pfn_hsa_system_get_info failed with %d", res);
    goto error;
  }

  INFO(NCCL_INIT, "ROCr version %d.%d", version_major, version_minor);

  //if (hsaDriverVersion < ROCR_DRIVER_MIN_VERSION) {
    // WARN("ROCr Driver version found is %d. Minimum requirement is %d", hsaDriverVersion, ROCR_DRIVER_MIN_VERSION);
    // Silently ignore version check mismatch for backwards compatibility
    //goto error;
  //}

  /* DMA-BUF support */
  res = pfn_hsa_system_get_info((hsa_system_info_t) 0x204, &dmaBufSupport);
  if (res != HSA_STATUS_SUCCESS || !dmaBufSupport) INFO(NCCL_ALL, "Current version of ROCm does not support dmabuf feature.");
  else {
    pfn_hsa_amd_portable_export_dmabuf = (PFN_hsa_amd_portable_export_dmabuf) dlsym(hsaLib, "hsa_amd_portable_export_dmabuf");
    if (pfn_hsa_amd_portable_export_dmabuf == NULL) {
      WARN("Failed to load ROCr missing symbol hsa_amd_portable_export_dmabuf");
      goto error;
    } 
    else {
      struct utsname utsname;
      FILE *fp = NULL;
      char kernel_opt1[28] = "CONFIG_DMABUF_MOVE_NOTIFY=y";
      char kernel_opt2[20] = "CONFIG_PCI_P2PDMA=y";
      char kernel_conf_file[128];
      char buf[256];
      int found_opt1 = 0;
      int found_opt2 = 0;
      
      //check for kernel name exists
      if (uname(&utsname) == -1) INFO(NCCL_ALL,"Could not get kernel name");
      //format and store the kernel conf file location
      snprintf(kernel_conf_file, sizeof(kernel_conf_file), "/boot/config-%s", utsname.release);
      fp = fopen(kernel_conf_file, "r");
      if (fp == NULL) INFO(NCCL_ALL,"Could not open kernel conf file");
      //look for kernel_opt1 and kernel_opt2 in the conf file and check
      while (fgets(buf, sizeof(buf), fp) != NULL) {
        if (strstr(buf, kernel_opt1) != NULL) {
          found_opt1 = 1;
          INFO(NCCL_ALL,"CONFIG_DMABUF_MOVE_NOTIFY=y in /boot/config-%s", utsname.release);
        }
        if (strstr(buf, kernel_opt2) != NULL) {
          found_opt2 = 1;
          INFO(NCCL_ALL,"CONFIG_PCI_P2PDMA=y in /boot/config-%s", utsname.release);
        }
      }
      if (!found_opt1 || !found_opt2) {
        dmaBufSupport = 0;
        INFO(NCCL_ALL, "CONFIG_DMABUF_MOVE_NOTIFY and CONFIG_PCI_P2PDMA should be set for DMA_BUF in /boot/config-%s", utsname.release);
        INFO(NCCL_ALL, "DMA_BUF_SUPPORT Failed due to OS kernel support");
      }

      if(dmaBufSupport) INFO(NCCL_ALL, "DMA_BUF Support Enabled");
      else goto error;
    }
  }

  /*
   * Required to initialize the ROCr Driver.
   * Multiple calls of hsa_init() will return immediately
   * without making any relevant change
   */
  pfn_hsa_init();

  hsaState = hsaInitialized;
  return ncclSuccess;

error:
  hsaState = hsaError;
  return ncclSystemError;
}


