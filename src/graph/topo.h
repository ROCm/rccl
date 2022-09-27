/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "graph.h"
#include "core.h"

#define LOC_WIDTH 5000.0
#define SM60_NVLINK_WIDTH 18.0
#define SM70_NVLINK_WIDTH 22.0
#define SM80_NVLINK_WIDTH 22.0
#define SM86_NVLINK_WIDTH 12.0
#define PCI_WIDTH 12.0           // PCI Gen3 x16
#define QPI_WIDTH 6.0
#define SKL_QPI_WIDTH 9.0
#define ZPI_WIDTH 6.0
#define YONGFENG_ZPI_WIDTH 9.0
#define P9_WIDTH 32.0
#define ARM_WIDTH 6.0
#define NET_WIDTH 12.0           // 100Gbit
#define VEGA_XGMI_WIDTH 24.0
#define MI200_XGMI_WIDTH 36.0

// Intel CPU convert GPU P2P traffic into 64B PCI TLPs, so GPU
// to GPU traffic consumes more PCI bandwidth.
#define INTEL_P2P_OVERHEAD(speed) (speed*6/5)

#define NCCL_TOPO_NODE_TYPES 7
#define GPU 0
#define PCI 1
#define NVS 2
#define CPU 3 // Actually NUMA domains
#define NIC 4
#define NET 5
extern const char* topoNodeTypeStr[];

// We want link types and path types to match as much as possible
#define LINK_LOC 0
#define LINK_NVL 1
// Skipping 2 for PATH_NVB
#define LINK_PCI 3
// Skipping 4 for PATH_PXB
// Skipping 5 for PATH_PXN
// Skipping 6 for PATH_PHB
#define LINK_SYS 7
#define LINK_NET 8
extern const char* topoLinkTypeStr[];

// Local (myself)
#define PATH_LOC 0

// Connection traversing NVLink
#define PATH_NVL 1

// Connection through NVLink using an intermediate GPU
#define PATH_NVB 2

// Connection traversing at most a single PCIe bridge
#define PATH_PIX 3

// Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
#define PATH_PXB 4

// Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network send/recv operations.
#define PATH_PXN 5

// Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
#define PATH_PHB 6

// Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
#define PATH_SYS 7
#define PATH_DIS 7
extern const char* topoPathTypeStr[];

struct ncclTopoNode;
struct ncclTopoLink {
  int type;
  float width;
  struct ncclTopoNode* remNode;
};
#define NCCL_TOPO_MAX_LINKS 32
#define NCCL_TOPO_MAX_HOPS (NCCL_TOPO_MAX_NODES*NCCL_TOPO_NODE_TYPES)

struct ncclTopoLinkList {
  struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];
  int count;
  float width;
  int type;
};

#define NCCL_TOPO_CPU_INTEL_BDW 1
#define NCCL_TOPO_CPU_INTEL_SKL 2

#define NCCL_TOPO_UNDEF (-1)

#define RCCL_TOPO_CR8G      1
#define RCCL_TOPO_4P2H_ROME 2
#define RCCL_TOPO_GDR_ALL   4
#define RCCL_TOPO_16P1H     8
#define RCCL_TOPO_FORCE_INTRA 16

#define RCCL_TOPO_MAX_RANKS_PER_GPU 8
struct ncclTopoNode {
  int type;
  int64_t id;
  // Type specific data
  union {
    struct {
      int dev; // NVML dev number
      int rank[RCCL_TOPO_MAX_RANKS_PER_GPU];
      int nRanksPerGpu;
      int cudaCompCap;
      int gdrSupport;
      int gcn;
      hipDeviceArch_t arch;
    }gpu;
    struct {
      uint64_t asic;
      int port;
      float width;
      float latency;
      int gdrSupport;
      int collSupport;
      int maxChannels;
      int64_t busId;
    }net;
    struct {
      int arch;
      int vendor;
      int model;
      cpu_set_t affinity;
    }cpu;
    struct {
      uint64_t device;
    }pci;
  };
  int nlinks;
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];
  // Pre-computed paths to GPUs and NICs
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];
  // Used during search
  uint64_t used;
};

struct ncclTopoNodeSet {
  int count;
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];
};

struct ncclTopoSystem {
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
  float maxWidth;
  float totalWidth;
  int type;
  int nRanks;
  int netGdrLevel;
  int tuning;

  bool pivotA2AEnabled;
  int pivotA2ANumBiRings;
  bool ll128Enabled;
  float baseBw;
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int id);
ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float width);
ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system);
ncclResult_t ncclTopoLoadSystem(const char* xmlTopoFile, struct ncclTopoSystem* system);
ncclResult_t ncclTopoGetIntermediateRank(struct ncclTopoSystem* system, int rank, int netDev, int* intermediateRank);

ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem);
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode *xmlGraphs, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels);
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system, struct ncclXml *xml);

ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax);

static ncclResult_t ncclTopoIdToIndex(struct ncclTopoSystem* system, int type, int64_t id, int* index) {
  *index = -1;
  for (int i=0; i<system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *index = i;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

static ncclResult_t ncclTopoRankToIndex(struct ncclTopoSystem* system, int rank, int* index) {
  *index = -1;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    for (int j=0; j<system->nodes[GPU].nodes[i].gpu.nRanksPerGpu; j++ ) {
      if (system->nodes[GPU].nodes[i].gpu.rank[j] == rank) {
	*index = i;
	return ncclSuccess;
      }
    }
  }
  return ncclInternalError;
}

// Returns XGMI speed in GB/s
static float ncclTopoXGMISpeed(int gcn) {
  return gcn == 910 ? MI200_XGMI_WIDTH : VEGA_XGMI_WIDTH;
}

#define ncclGetKernelIndex(p_comm) \
  (((p_comm)->topo->ll128Enabled ? 1 : 0)*2 + ((p_comm)->hostDevComm.collTraceThread ? 1 : 0))

#endif
