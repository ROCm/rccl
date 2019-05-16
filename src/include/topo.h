/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "nccl.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <string>

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

static bool isEPYC() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  int needed = 2;
  static bool vendor_id = true, cpu_family = false, initialized = false;
  if (initialized) return (vendor_id && cpu_family);
  while (std::getline(cpuinfo, line)) {
    if (line.compare(0, 9, "vendor_id") == 0) {
      if(line.find("AuthenticAMD") == std::string::npos)
        vendor_id = false;
      needed --;
    }
    if (line.compare(0, 10, "cpu family") == 0) {
      std::string family_str = line.substr(line.find(": ") + 2);
      if (std::stoi(family_str) >= 23)
        cpu_family = true;
      needed --;
    }
    if (!needed)
      break;
  }
  initialized = true;
  return (vendor_id && cpu_family);
}

static ncclResult_t getCudaPath(int cudaDev, char** path) {
  char busId[BUSID_SIZE];
  CUDACHECK(hipDeviceGetPCIBusId(busId, BUSID_SIZE, cudaDev));
  for (int i=0; i<BUSID_SIZE; i++) busId[i] = tolower(busId[i]);
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  memcpy(busPath+sizeof("/sys/class/pci_bus/")-1, busId, BUSID_REDUCED_SIZE-1);
  memcpy(busPath+sizeof("/sys/class/pci_bus/0000:00/../../")-1, busId, BUSID_SIZE-1);
  *path = realpath(busPath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", busPath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

enum ncclPathDist {
  PATH_PIX = 0,
  PATH_PXB = 1,
  PATH_PHB = 2,
  PATH_SOC = 3
};

static const char* pathDists[] = { "PIX", "PXB", "PHB", "SOC" };

static int pciDistance(char* path1, char* path2) {
  int score = 0;
  int depth = 0;
  int same = 1;
  for (int i=0; i<strlen(path1); i++) {
    if (path1[i] != path2[i]) same = 0;
    if (path1[i] == '/') {
      depth++;
      if (same == 1) score++;
    }
  }
  if (isEPYC() && score <= 3) return PATH_PHB;
  if (score <= 3) return PATH_SOC;
  if (score == 4) return PATH_PHB;
  if (score == depth-1) return PATH_PIX;
  return PATH_PXB;
}

#endif
