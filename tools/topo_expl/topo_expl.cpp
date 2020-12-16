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

#include "nccl.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "graph.h"
#include "argcheck.h"
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <iostream>
#include <cstring>
#include "model.h"
#include "utils.h"
#include "topo.h"

NodeModel *node_model;

char* getCmdOption(char ** begin, char ** end, const std::string & option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

typedef struct NodeModelDesc {
    int         num_nodes;
    const char *filename;
    const char *description;
} NodeModelDesc;

NodeModelDesc model_descs[] = {
  {1, "topo_4p1h.xml",          "single node VEGA20 4P1H"},
  {1, "topo_4p1h_1.xml",        "single node VEGA20 4P1H Alt. Model"},
  {1, "topo_4p2h.xml",          "single node VEGA20 4P2H"},
  {1, "topo_4p3l.xml",          "single node gfx908 4P3L"},
  {1, "topo_8p6l.xml",          "single node gfx908 8P6L"},
  {1, "topo_8p_pcie.xml",       "single node 8 VEGA20 PCIe"},
  {4, "topo_8p_pcie.xml",       "4 nodes with 8 GPUs PCIe 1 NIC"},
  {4, "topo_8p_pcie_1.xml",     "4 nodes with 8 GPUs PCIe 1 NIC 2nd PLX Bridge"},
  {4, "topo_8p_pcie_2nic.xml",  "4 nodes with 8 GPUs PCIe 2 NIC"},
  {2, "topo_4p1h.xml",          "2 nodes VEGA20 4P1H"},
  {4, "topo_4p2h.xml",          "4 nodes with 8 VEGA20 GPUs XGMI 4P2H 1 NIC"},
  {4, "topo_4p2h_1.xml",        "4 nodes with 8 VEGA20 GPUs XGMI 4P2H 1 NIC 2nd Hive"},
  {4, "topo_4p2h_2nic.xml",     "4 nodes with 8 VEGA20 GPUs XGMI 4P2H 2 NIC"},
  {1, "topo_8p_rome.xml",       "single node 8 VEGA20 Rome"},
  {4, "topo_8p6l.xml",          "4 nodes gfx908 8P6L 1 NIC 2nd Hive"},
  {4, "topo_8p6l_1nic.xml",     "4 nodes gfx908 8P6L 1 NIC"},
  {4, "topo_8p6l_2nic.xml",     "4 nodes gfx908 8P6L 2 NICs"},
  {4, "topo_8p6l_3nic.xml",     "4 nodes gfx908 8P6L 3 NICs"},
  {4, "topo_8p6l_4nic.xml",     "4 nodes gfx908 8P6L 4 NICs"},
  {4, "topo_8p6l_5nic.xml",     "4 nodes gfx908 8P6L 5 NICs"},
  {4, "topo_8p6l_6nic.xml",     "4 nodes gfx908 8P6L 6 NICs"},
  {1, "topo_8p_rome_n2.xml",    "single node 8 VEGA20 Rome NPS=2"},
  {4, "topo_8p_rome_n2.xml",    "4 nodes 8 VEGA20 Rome NPS=2"},
  {1, "topo_8p_rome_n2_1.xml",  "single node 8 VEGA20 Rome NPS=2 Alt. Model"},
  {1, "topo_8p_ts1.xml",        "single node 8 VEGA20 TS1"},
  {4, "topo_8p_ts1.xml",        "4 nodes 8 VEGA20 TS1"},
  {1, "topo_8p_ts1_1.xml",      "single node 8 VEGA20 TS1 Alt. Model"},
  {4, "topo_8p_ts1_1.xml",      "4 nodes 8 VEGA20 TS1 Alt. Model"},
  {1, "topo_4p3l_2h.xml",       "single node 8 gfx908 Rome"},
  {4, "topo_4p3l_2h.xml",       "4 nodes 8 gfx908 Rome"},
  {1, "topo_8p_ts1_n4.xml",     "single node 8 VEGA20 TS1 NPS=4"},
  {4, "topo_8p_ts1_n4.xml",     "4 nodes 8 VEGA20 TS1 NPS=4"},
  {1, "topo_8p_ts1_n4_1.xml",   "single node 8 VEGA20 TS1 NPS=4 Alt. Model"},
  {4, "topo_8p_ts1_n4_1.xml",   "4 nodes 8 VEGA20 TS1 NPS=4 Alt. Model"},
  {1, "topo_4p3l_ia.xml",       "single node 8 gfx908"},
  {4, "topo_4p3l_ia.xml",       "4 nodes 8 gfx908"},
  {4, "topo_8p_rome_n2_2.xml",  "4 nodes 8 VEGA20 Rome NPS=2 Alt. Model 2 NET/IF"},
  {4, "topo_8p_ts1_n4_2.xml",   "4 nodes 8 VEGA20 TS1 NPS=4 3 NET/IF"},
  {1, "topo_8p_rome_n4.xml",    "single node 8 VEGA20 Rome NPS=4"},
  {1, "topo_4p3l_n2.xml",       "single node 8 gfx908 Rome"},
  {4, "topo_4p3l_n2.xml",       "4 nodes 8 gfx908 Rome"},
  {1, "topo_4p3l_n4.xml",       "single node 8 gfx908 Rome NPS=4"},
  {4, "topo_4p3l_n4.xml",       "4 nodes 8 gfx908 Rome NPS=4"},
  {1, "topo_4p3l_n2_1.xml",     "single node 8 gfx908 Rome"},
  {4, "topo_4p3l_n2_1.xml",     "4 nodes 8 gfx908 Rome"},
  {1, "topo_8p_rome_n4_1.xml",  "single node 8 gfx908 Rome NPS=4"},
  {4, "topo_8p_rome_n4_1.xml",  "4 nodes node 8 gfx908 Rome NPS=4"},
  {2, "topo_8p_rome_pcie.xml",  "2 nodes node 8 VEGA20 PCIe"},
};

int main(int argc,char* argv[])
{
  struct ncclComm *comm;
  const int num_models = sizeof(model_descs) / sizeof(*model_descs);

  if (!cmdOptionExists(argv, argv + argc, "-m")) {
    printf("Usage: ./topo_expl -m model_id\n");
    printf("List of model_id:\n");
    for (int i = 0; i < num_models; i++)
      printf("  %d: %s\n", i, model_descs[i].description);
    exit(0);
  }

  int model_id = 0;
  char *mi = getCmdOption(argv, argv + argc, "-m");
  if (mi)
    model_id = atol(mi);

  if (model_id >= num_models) {
      printf("Invalid model_id %d\n", model_id);
      exit(0);
  }

  NetworkModel network;
  NodeModel* node;

  NodeModelDesc *desc = &model_descs[model_id];
  for (int i=0; i<desc->num_nodes; i++) {
      node = new NodeModel(desc->filename);
      network.AddNode(node);
  }

  printf("Generating topology using %d: %s\n", model_id, desc->description);

  int nranks = network.GetNRanks();
  int nnodes = network.GetNNodes();

  printf("nnodes = %d, nranks = %d\n", nnodes, nranks);
  for (int i = 0; i < nranks; i++) {
    node_model = network.GetNode(i);
    assert(node_model!=0);
    printf("Rank %d: node %d cudaDev %d GPU busId %lx\n", i, node_model->nodeId,
      node_model->rankToCudaDev(i), node_model->getGpuBusId(i));
  }

  NCCLCHECK(ncclCalloc(&comm, nranks));

  struct allGather1Data_t *allGather1Data;
  NCCLCHECK(ncclCalloc(&allGather1Data, nranks));

  struct allGather3Data_t *allGather3Data;
  NCCLCHECK(ncclCalloc(&allGather3Data, nranks));

  for (int i = 0; i < nranks; i++) {
    comm[i].rank = i;
    comm[i].nRanks = nranks;
    comm[i].p2plist.count=0;
    NCCLCHECK(ncclCalloc(&comm[i].p2plist.connect.recv, MAXCHANNELS*comm->nRanks));
    NCCLCHECK(ncclCalloc(&comm[i].p2plist.connect.send, MAXCHANNELS*comm->nRanks));
    node_model = network.GetNode(i);
    assert(node_model!=0);
    comm[i].topo = node_model->getSystem(i);
    bootstrapAllGather(&comm[i], allGather1Data);
  }

  struct ncclTopoGraph *treeGraph, *ringGraph, *collNetGraph;
  treeGraph = (struct ncclTopoGraph *)malloc(sizeof(struct ncclTopoGraph)*nranks);
  ringGraph = (struct ncclTopoGraph *)malloc(sizeof(struct ncclTopoGraph)*nranks);
  collNetGraph = (struct ncclTopoGraph *)malloc(sizeof(struct ncclTopoGraph)*nranks);
  if (!treeGraph || !ringGraph || !collNetGraph) {
    printf("Failed to allocate memory for graphs\n");
    return -1;
  }

  for (int i = 0; i < nranks; i++) {
    node_model = network.GetNode(i);
    assert(node_model!=0);
    initTransportsRank_1(&comm[i], allGather1Data, allGather3Data, treeGraph[i], ringGraph[i], collNetGraph[i]);
  }

  for (int i = 0; i < nranks; i++) {
    node_model = network.GetNode(i);
    assert(node_model!=0);
    initTransportsRank_3(&comm[i], allGather3Data, treeGraph[i], ringGraph[i], collNetGraph[i]);
  }

  free(treeGraph);
  free(ringGraph);
  free(collNetGraph);
  free(allGather3Data);
  free(allGather1Data);

  free(comm);
  printf("Done generating topology using %d: %s\n", model_id, desc->description);

  return 0;
}
