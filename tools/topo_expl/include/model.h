/*
Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef MODEL_H_
#define MODEL_H_

class CpuDevices {
private:
  char *cpuName;
  int interCpuWidth;
  int cpuPciWidth;
  int p2pPciWidth;

public:
  CpuDevices(const char *cpuname, const int intercpuwidth, const int cpupciwidth, const int p2ppciwidth) :
    cpuName((char *)cpuname), interCpuWidth(intercpuwidth), cpuPciWidth(cpupciwidth), p2pPciWidth(p2ppciwidth) {}

  CpuDevices() : cpuName(0), interCpuWidth(0), cpuPciWidth(0), p2pPciWidth(0) {}

  ncclResult_t getCpuWidths(char* name, int* interCpu, int* cpuPci, int* p2pPci) {
    strcpy(name, cpuName);
    *interCpu = interCpuWidth;
    *cpuPci = cpuPciWidth;
    *p2pPci = p2pPciWidth;
    return ncclSuccess;
  }
};

class GpuDevices {
private:
  int nGpus;
  uint64_t *busIds;
  char **gpuPciPaths;
  int *gpuNumaIds;
  int *connMatrix;

public:
  GpuDevices(const int ngpus, const uint64_t *busids, const char **gpupcipaths, const int *gpunumaids, const int *connmatrix) :
    nGpus(ngpus), busIds((uint64_t *)busids), gpuPciPaths((char **)gpupcipaths), gpuNumaIds((int *)gpunumaids), connMatrix((int *)connmatrix) {}

  GpuDevices () : nGpus(0), busIds(0), gpuPciPaths(0), gpuNumaIds(0), connMatrix(0) {}

  int getnDevs() { return nGpus; }

  uint64_t getBusId(int dev) { return busIds[dev]; }

  ncclResult_t getPciPath(char* busId, char** path) {
    char tempBusId[] = "0000:00:00.0";
    *path = (char *)malloc(PATH_MAX);
    int i;
    for (i = 0; i < nGpus; i++) {
      NCCLCHECK(int64ToBusId(busIds[i], tempBusId));
      if (strcmp(busId, tempBusId) == 0)
        break;
    }
    if (i < nGpus)
      strcpy(*path, gpuPciPaths[i]);
    else {
      WARN("Could not find real path of %s", busId);
      return ncclSystemError;
    }
    return ncclSuccess;
  }

  int p2pCanConnect(int device1, int device2) {
    // connection matrix are 8 GPUs
    int dist = connMatrix[device1*8+device2];
    if (dist == 255)
      return 0;
    //if (dist%15 == 0 && dist/15 != 1) {
    //  return 0;
    //}
    return 1;
  };

  hipError_t getLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype, uint32_t* hopcount) {
    // connection matrix are 8 GPUs
    int dist = connMatrix[device1*8+device2];

    if (dist%15 == 0) {
      *linktype = 4;
      *hopcount = dist/15;
    }
    else if (dist%20 == 0) {
      *linktype = 2;
      *hopcount = dist/20;
    }
    else if (dist%36 == 0) {
      *linktype = 1;
      *hopcount = dist/36;
    }
    return hipSuccess;
  }

  virtual int getNumaId(char *path) {
    int n;
    // search for all GPUs
    for (n = 0; n < nGpus; n++)
      if (strcmp(path, gpuPciPaths[n]) == 0)
        break;
    if (n < nGpus)
      return gpuNumaIds[n];
    return -1;
  }
};

class NetDevices {
private:
  int nNetDevs;
  char **netPciPaths;
  uint64_t *netGuids;    // IB ports on same card share the same GUID
  int *netNumaIds;

public:
  NetDevices(const int nnetdevs, const char **netpcipaths, const uint64_t *netguids, const int *netnumaids) :
    nNetDevs(nnetdevs), netPciPaths((char **)netpcipaths), netGuids((uint64_t *)netguids), netNumaIds((int *)netnumaids) {}

  NetDevices() : nNetDevs(0), netPciPaths(0), netGuids(0), netNumaIds(0) {}

  int getnDevs() { return nNetDevs; }

  ncclResult_t getPciPath(int dev, char** path) {
    *path = (char *)malloc(PATH_MAX);
    if (dev < nNetDevs)
      strcpy(*path, netPciPaths[dev]);
    else {
      WARN("Could not find real path of %d", dev);
      return ncclSystemError;
    }
    return ncclSuccess;
  }

  virtual int getNumaId(char *path) {
    int n;
    // search for all NICs
    for (n = 0; n < nNetDevs; n++)
      if (strcmp(path, netPciPaths[n]) == 0)
        break;
    if (n < nNetDevs)
      return netNumaIds[n];
    return -1;
  }

  uint64_t getIbGuid(char* path) {
    int n;
    for (n = 0; n < nNetDevs; n++)
      if (strcmp(path, netPciPaths[n]) == 0)
        break;
    if (n < nNetDevs)
      return netGuids[n];
    WARN("Invalid IB path %s", path);
    return 0;
  }
};

class NodeModel {
private:
  CpuDevices cpus;
  GpuDevices gpus;
  NetDevices netdevs;

public:
  int nodeId;
  int currRank;
  int firstRank;
  uint64_t hostHash;  // auto-generated
  uint64_t pidHash;   // auto-generated
  char description[256];

  int rankToCudaDev(int rank) { return rank - firstRank; }

  int getnGpus() { return gpus.getnDevs(); }

  int getnNetDevs() { return netdevs.getnDevs(); }

  ncclResult_t getGpuPciPath(char* busId, char** path) {
    return gpus.getPciPath(busId, path);
  }

  ncclResult_t getNetPciPath(int dev, char** path) {

    return netdevs.getPciPath(dev, path);
  }

  uint64_t getGpuBusId(int dev) {
    return gpus.getBusId(dev);
  }

  int p2pCanConnect(int device1, int device2) { return gpus.p2pCanConnect(device1, device2); }

  hipError_t getLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype, uint32_t* hopcount) {
    return gpus.getLinkTypeAndHopCount(device1, device2, linktype, hopcount);
  }

  uint64_t getIbGuid(char* path) {
    return netdevs.getIbGuid(path);
  }

  int shmCanConnect(int device1, int device2) { return 1; }
  int netCanConnect(int device1, int device2) { return 1; }

  virtual int getNumaId(char *path) {
    int numa = gpus.getNumaId(path);
    if (numa != -1) return numa;
    numa = netdevs.getNumaId(path);
    if (numa != -1) return numa;
    WARN("Invalid path %s for getNumaId", path);
    return 0;
  }

  virtual ncclResult_t getCpuWidths(char* name, int* interCpu, int* cpuPci, int* p2pPci) {
    return cpus.getCpuWidths(name, interCpu, cpuPci, p2pPci);
  }

  NodeModel(CpuDevices cpu, GpuDevices gpu, NetDevices net, const char *desc) :
    cpus(cpu), gpus(gpu), netdevs(net) {
      strncpy(description, desc, 256);
  }

  NodeModel() {}

  ~NodeModel() {}
};

class NetworkModel {
private:
  int nNodes;
  int nRanks;
  NodeModel nodes[NCCL_TOPO_MAX_NODES];

public:
  void AddNode(NodeModel node) {
    nodes[nNodes] = node;
    nodes[nNodes].nodeId = nNodes;
    nodes[nNodes].firstRank = nRanks;
    nodes[nNodes].hostHash = ((uint64_t)rand() << 32) | rand();
    nodes[nNodes].pidHash = ((uint64_t)rand() << 32) | rand();
    nNodes++;
    nRanks += node.getnGpus();
  }

  int GetNNodes() { return nNodes; }

  int GetNRanks() { return nRanks; }

  NodeModel* GetNode(int rank) {
    int node_id;

    if(rank < 0 || rank >= nRanks)
      return 0;

    for(node_id = nNodes-1; node_id >= 0; node_id--)
      if(rank >= nodes[node_id].firstRank) break;

    if (node_id >= 0) {
      nodes[node_id].currRank = rank;
      return nodes+node_id;
    }
    else
      return 0;
  }

  NetworkModel() : nNodes(0), nRanks(0) {}
};


const static uint64_t busIds_8[] = { 0x1d000, 0x20000, 0x23000, 0x26000, 0x3f000, 0x43000, 0x46000, 0x49000 };

const static char* gpuPciPaths_8[] = {
  "/sys/devices/pci0000:17/0000:17:00.0/0000:18:00.0/0000:19:08.0/0000:1b:00.0/0000:1c:00.0/0000:1d:00.0",
  "/sys/devices/pci0000:17/0000:17:00.0/0000:18:00.0/0000:19:0c.0/0000:1e:00.0/0000:1f:00.0/0000:20:00.0",
  "/sys/devices/pci0000:17/0000:17:00.0/0000:18:00.0/0000:19:10.0/0000:21:00.0/0000:22:00.0/0000:23:00.0",
  "/sys/devices/pci0000:17/0000:17:00.0/0000:18:00.0/0000:19:14.0/0000:24:00.0/0000:25:00.0/0000:26:00.0",
  "/sys/devices/pci0000:3a/0000:3a:00.0/0000:3b:00.0/0000:3c:04.0/0000:3d:00.0/0000:3e:00.0/0000:3f:00.0",
  "/sys/devices/pci0000:3a/0000:3a:00.0/0000:3b:00.0/0000:3c:0c.0/0000:41:00.0/0000:42:00.0/0000:43:00.0",
  "/sys/devices/pci0000:3a/0000:3a:00.0/0000:3b:00.0/0000:3c:10.0/0000:44:00.0/0000:45:00.0/0000:46:00.0",
  "/sys/devices/pci0000:3a/0000:3a:00.0/0000:3b:00.0/0000:3c:14.0/0000:47:00.0/0000:48:00.0/0000:49:00.0",
};

const static int gpuPciNumaIds_8[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

const static char* netPciPaths_1[] = {
  "/sys/devices/pci0000:17/0000:17:00.0/0000:18:00.0/0000:19:04.0/0000:1a:00.0",
};

const static char* netPciPaths_1_1[] = {
  "/sys/devices/pci0000:3a/0000:3a:00.0/0000:3b:00.0/0000:3c:08.0/0000:4c:00.0",
};

const static uint64_t netGuids_1[] = {
  0xb8599f030007053aL,
};

const static int netPciNumaIds_1[] = { 0 };

const static char* netPciPaths_2[] = {
  "/sys/devices/pci0000:17/0000:17:00.0/0000:18:00.0/0000:19:04.0/0000:1a:00.0",
  "/sys/devices/pci0000:3a/0000:3a:00.0/0000:3b:00.0/0000:3c:08.0/0000:4c:00.0",
};

const static uint64_t netGuids_2[] = {
  0xb8599f030007053aL,
  0x506b4b030027bbf2L,
};

const static int netPciNumaIds_2[] = { 0, 0 };

const static uint64_t rome_busIds_8[] = { 0x63000, 0x23000, 0x26000, 0x03000, 0xe3000, 0xc3000, 0xc6000, 0xa3000 };

const static char* rome_gpuPciPaths_8[] = {
  "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0/0000:62:00.0/0000:63:00.0",
  "/sys/devices/pci0000:20/0000:20:01.1/0000:21:00.0/0000:22:00.0/0000:23:00.0",
  "/sys/devices/pci0000:20/0000:20:03.1/0000:24:00.0/0000:25:00.0/0000:26:00.0",
  "/sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0/0000:03:00.0",
  "/sys/devices/pci0000:e0/0000:e0:03.1/0000:e1:00.0/0000:e2:00.0/0000:e3:00.0",
  "/sys/devices/pci0000:c0/0000:c0:01.1/0000:c1:00.0/0000:c2:00.0/0000:c3:00.0",
  "/sys/devices/pci0000:c0/0000:c0:03.1/0000:c4:00.0/0000:c5:00.0/0000:c6:00.0",
  "/sys/devices/pci0000:a0/0000:a0:03.1/0000:a1:00.0/0000:a2:00.0/0000:a3:00.0",
};

const static int rome_gpuPciNumaIds_8[] = { 0, 0, 0, 0, 4, 4, 4, 4 };

const static char* rome_netPciPaths_1[] = {
  "/sys/devices/pci0000:40/0000:40:01.1/0000:41:00.0",
};

const static uint64_t rome_netGuids_1[] = {
  0xb8599f030007053aL,
};

const static int rom_netPciNumaIds_1[] = { 0 };

const static char* rome_netPciPaths_2[] = {
  "/sys/devices/pci0000:40/0000:40:01.1/0000:41:00.0",
  "/sys/devices/pci0000:80/0000:80:01.1/0000:81:00.0",
};

const static uint64_t rome_netGuids_2[] = {
  0xb8599f030007053aL,
  0x506b4b030027bbf2L,
};

const static int rom_netPciNumaIds_2[] = { 0, 4 };

const int conn_mat_pcie[64] = {
  0 , 40, 40, 40, 40, 40, 40, 40,
  40, 0 , 40, 40, 40, 40, 40, 40,
  40, 40, 0 , 40, 40, 40, 40, 40,
  40, 40, 40, 0 , 40, 40, 40, 40,
  40, 40, 40, 40, 0 , 40, 40, 40,
  40, 40, 40, 40, 40, 0 , 40, 40,
  40, 40, 40, 40, 40, 40, 0 , 40,
  40, 40, 40, 40, 40, 40, 40, 0 ,
};

const int conn_mat_4p2h[64] = {
  0 , 15, 15, 30, 40, 40, 40, 40,
  15, 0 , 30, 15, 40, 40, 40, 40,
  15, 30, 0 , 15, 40, 40, 40, 40,
  30, 15, 15, 0 , 40, 40, 40, 40,
  40, 40, 40, 40, 0 , 15, 15, 30,
  40, 40, 40, 40, 15, 0 , 30, 15,
  40, 40, 40, 40, 15, 30, 0 , 15,
  40, 40, 40, 40, 30, 15, 15, 0 ,
};

const int conn_mat_8p6l[64] = {
  0 , 15, 15, 15, 15, 30, 15, 15,
  15, 0 , 15, 15, 30, 15, 15, 15,
  15, 15, 0 , 15, 15, 15, 15, 30,
  15, 15, 15, 0 , 15, 15, 30, 15,
  15, 30, 15, 15, 0 , 15, 15, 15,
  30, 15, 15, 15, 15, 0 , 15, 15,
  15, 15, 15, 30, 15, 15, 0 , 15,
  15, 15, 30, 15, 15, 15, 15, 0 ,
};

const int conn_mat_8p6l_1[64] = {
 0 , 15, 15, 30, 15, 15, 15, 15,
 15, 0 , 30, 15, 15, 15, 15, 15,
 15, 30, 0 , 15, 15, 15, 15, 15,
 30, 15, 15, 0 , 15, 15, 15, 15,
 15, 15, 15, 15, 0 , 15, 15, 30,
 15, 15, 15, 15, 15, 0 , 30, 15,
 15, 15, 15, 15, 15, 30, 0 , 15,
 15, 15, 15, 15, 30, 15, 15, 0 ,
};

const int conn_mat_rome[64] = {
  0 , 40, 40, 40, 72, 72, 72, 72,
  40, 0 , 40, 40, 72, 72, 72, 72,
  40, 40, 0 , 40, 72, 72, 72, 72,
  40, 40, 40, 0 , 72, 72, 72, 72,
  72, 72, 72, 72, 0 , 40, 40, 40,
  72, 72, 72, 72, 40, 0 , 40, 40,
  72, 72, 72, 72, 40, 40, 0 , 40,
  72, 72, 72, 72, 40, 40, 40, 0 ,
};

#endif