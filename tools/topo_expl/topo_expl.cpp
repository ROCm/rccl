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

const char *model_descriptions[] = {
  "single node VEGA20 4P1H",
  "single node VEGA20 4P1H Alt. Model",
  "single node VEGA20 4P2H",
  "single node gfx908 4P3L",
  "single node gfx908 8P6L",
  "single node 8 VEGA20 PCIe",
  "4 nodes with 8 GPUs PCIe 1 NIC",
  "4 nodes with 8 GPUs PCIe 1 NIC 2nd PLX Bridge",
  "4 nodes with 8 GPUs PCIe 2 NIC",
  "2 nodes VEGA20 4P1H",
  "4 nodes with 8 VEGA20 GPUs XGMI 4P2H 1 NIC",
  "4 nodes with 8 VEGA20 GPUs XGMI 4P2H 1 NIC 2nd Hive",
  "4 nodes with 8 VEGA20 GPUs XGMI 4P2H 2 NIC",
  NULL,
};

int main(int argc,char* argv[])
{
  struct ncclComm *comm;

  if (!cmdOptionExists(argv, argv + argc, "-m")) {
    printf("Usage: ./topo_expl -m model_id\n");
    printf("List of model_id:\n");
    for (int i = 0; model_descriptions[i] != NULL; i++)
      printf("  %d: %s\n", i, model_descriptions[i]);
    exit(0);
  }

  int model_id = 0;
  char *mi = getCmdOption(argv, argv + argc, "-m");
  if (mi)
    model_id = atol(mi);

  NetworkModel network;
  NodeModel* node;

  switch(model_id) {
    case 0:
      node = new NodeModel("topo_4p1h.xml");
      network.AddNode(node);
      break;
    case 1:
      node = new NodeModel("topo_4p1h_1.xml");
      network.AddNode(node);
      break;
    case 2:
      node = new NodeModel("topo_4p2h.xml");
      network.AddNode(node);
      break;
    case 3:
      node = new NodeModel("topo_4p3l.xml");
      network.AddNode(node);
      break;
    case 4:
      node = new NodeModel("topo_8p6l.xml");
      network.AddNode(node);
      break;
    case 5:
      node = new NodeModel("topo_8p_pcie.xml");
      network.AddNode(node);
      break;
    case 6:
      for (int i=0; i<4; i++) {
        node = new NodeModel("topo_8p_pcie.xml");
        network.AddNode(node);
      }
      break;
    case 7:
      for (int i=0; i<4; i++) {
        node = new NodeModel("topo_8p_pcie_1.xml");
        network.AddNode(node);
      }
      break;
    case 8:
      for (int i=0; i<4; i++) {
        node = new NodeModel("topo_8p_pcie_2nic.xml");
        network.AddNode(node);
      }
      break;
    case 9:
      for (int i=0; i<2; i++) {
        node = new NodeModel("topo_4p1h.xml");
        network.AddNode(node);
      }
      break;
    case 10:
      for (int i=0; i<4; i++) {
        node = new NodeModel("topo_4p2h.xml");
        network.AddNode(node);
      }
      break;
    case 11:
      for (int i=0; i<4; i++) {
        node = new NodeModel("topo_4p2h_1.xml");
        network.AddNode(node);
      }
      break;
    case 12:
      for (int i=0; i<4; i++) {
        node = new NodeModel("topo_4p2h_2nic.xml");
        network.AddNode(node);
      }
      break;
    default:
      printf("Invalid model_id %d\n", model_id);
      exit(0);
  }

  printf("Generating topology using %d: %s\n", model_id, model_descriptions[model_id]);

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
    node_model = network.GetNode(i);
    assert(node_model!=0);
    comm[i].topo = node_model->getSystem(i);
    bootstrapAllGather(&comm[i], allGather1Data);
  }

  struct ncclTopoGraph treeGraph, ringGraph, collNetGraph;

  for (int i = 0; i < nranks; i++) {
    node_model = network.GetNode(i);
    assert(node_model!=0);
    initTransportsRank_1(&comm[i], allGather1Data, allGather3Data, treeGraph, ringGraph, collNetGraph);
  }

  for (int i = 0; i < nranks; i++) {
    node_model = network.GetNode(i);
    assert(node_model!=0);
    initTransportsRank_3(&comm[i], allGather3Data, treeGraph, ringGraph, collNetGraph);
  }

  free(allGather3Data);
  free(allGather1Data);

  free(comm);
  printf("Done generating topology using %d: %s\n", model_id, model_descriptions[model_id]);

  return 0;
}