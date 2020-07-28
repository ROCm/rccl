/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "xml.h"
#include <math.h>

// Initialize system->maxWidth. This is the per-channel (i.e. per-SM)
// max speed.
static float getMaxWidth(struct ncclTopoSystem* system, struct ncclTopoNode* gpu, int type) {
  float maxWidth = 0.0;
  for (int i=0; i<system->nodes[type].count; i++) {
    struct ncclTopoLinkList* path = gpu->paths[type]+i;
    float width = path->width;
    if (path->count == 0) continue;
    maxWidth = std::max(maxWidth, width);
  }
  return maxWidth;
}
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
  system->maxWidth = 0.0;
  system->type = 0;
  int inter = system->nodes[NET].count;
  if (inter == 0 && system->nodes[GPU].count == 1) {
    system->maxWidth = LOC_WIDTH;
    return ncclSuccess;
  }
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    system->maxWidth = std::max(system->maxWidth, getMaxWidth(system, gpu, inter ? NET : GPU));
  }
  return ncclSuccess;
}

static ncclResult_t findRevLink(struct ncclTopoNode* node1, struct ncclTopoNode* node2, struct ncclTopoLink** revLink) {
  for (int l=0; l<node2->nlinks; l++) {
    struct ncclTopoLink* link = node2->links+l;
    if (link->remNode == node1) {
      *revLink = link;
      return ncclSuccess;
    }
  }
  WARN("Could not find rev link for %d/%d -> %d/%d\n", node1->type, node1->id, node2->type, node2->id);
  return ncclInternalError;
}

// This is unfortunately needed since manipulating floats often results in rounding errors.
#define SUB_ROUND(a, b) (a = roundf((a-b)*1000)/1000)

static ncclResult_t followPath(struct ncclTopoLinkList* path, struct ncclTopoNode* start, int maxSteps, float speed, int* steps) {
  float pciSpeed = speed;
  for (int step=0; step<path->count; step++) {
    struct ncclTopoNode* node = path->list[step]->remNode;
    if (node->type == CPU) {
      // Account for P2P inefficiency through Intel CPU RC
      if (path->type == PATH_PHB && start->type == GPU &&
          node->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 &&
          node->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
        pciSpeed = INTEL_P2P_OVERHEAD(speed);
      }
    }
  }

  struct ncclTopoNode* node = start;
  for (int step=0; step<maxSteps; step++) {
    struct ncclTopoLink* link = path->list[step];
    struct ncclTopoLink* revLink = NULL;
    float fwSpeed = link->type == LINK_PCI ? pciSpeed : speed;
    float revSpeed = 0;
    if (link->remNode->type == GPU && link->remNode->gpu.cudaCompCap < 80 && start->type != GPU) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, &revLink));
      revSpeed += fwSpeed/8;
    }
    if (link->remNode->type == CPU && link->type == LINK_NVL) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, &revLink));
      revSpeed += fwSpeed;
    }
    if (link->width < fwSpeed || (revSpeed && revLink->width < revSpeed)) { *steps = step; return ncclSuccess; }
    SUB_ROUND(link->width, fwSpeed);
    if (revSpeed) SUB_ROUND(revLink->width, revSpeed);
    node = link->remNode;
  }
  *steps = maxSteps;
  return ncclSuccess;
}

// Try to go from node type1/index1 to no type2/index2. mult indicates whether we are counting the bandwidth (1) or undoing (-1).
static ncclResult_t ncclTopoFollowPath(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int type1, int index1, int type2, int index2, int mult, struct ncclTopoNode** node) {
  // First handle easy cases
  *node = system->nodes[type2].nodes+index2;
  if (type1 == -1) return ncclSuccess;
  struct ncclTopoNode* node1 = system->nodes[type1].nodes+index1;
  struct ncclTopoLinkList* path = node1->paths[type2]+index2;
  if (path->count == 0 ) return ncclSuccess;

  // Now check link type
  *node = NULL;
  int intra = type1 == GPU && type2 == GPU;
  float speed = intra ? graph->speedIntra : graph->speedInter;
  int type = intra ? graph->typeIntra : graph->typeInter;

  if (mult == 1 && (path->type > type)) return ncclSuccess;

  speed *= mult;

  // Check there is enough bandwidth on paths.
  int step = 0;
  NCCLCHECK(followPath(path, node1, path->count, speed, &step));
  if (step < path->count) goto rewind;

  // Enough bandwidth : return destination node.
  graph->nHops += mult*path->count;
  *node = system->nodes[type2].nodes+index2;
  return ncclSuccess;

rewind:
  // Not enough bandwidth : rewind and exit.
  NCCLCHECK(followPath(path, node1, step, -speed, &step));
  return ncclSuccess;
}

static int gpuPciWidth(struct ncclTopoNode* gpu) {
  for (int l=0; l<gpu->nlinks; l++) {
    struct ncclTopoLink* gpuLink = gpu->links+l;
    if (gpuLink->type != LINK_PCI) continue;
    struct ncclTopoNode* pci = gpuLink->remNode;
    for (int l=0; l<pci->nlinks; l++) {
      struct ncclTopoLink* pciLink = pci->links+l;
      if (pciLink->remNode != gpu) continue;
      return std::min(gpuLink->width, pciLink->width);
    }
  }
  return -1;
}

/* Choose the order in which we try next GPUs. This is critical for the search
   to quickly converge to the best solution even if it eventually times out. */
struct ncclGpuScore {
  int g;             // Retain the index
  int startIndex;    // Least important
  int intraNhops;
  int intraWidth;
  int interNhops;
  int interPciWidth;
  int interWidth;    // Most important
};

static int cmpScore(const void * g1, const void * g2) {
   struct ncclGpuScore *s1 = (struct ncclGpuScore*)g1;
   struct ncclGpuScore *s2 = (struct ncclGpuScore*)g2;
   int d;
   if ((d = (s2->interWidth - s1->interWidth))) return d;
   if ((d = (s2->interPciWidth - s1->interPciWidth))) return d;
   if ((d = (s1->interNhops - s2->interNhops))) return d;
   if ((d = (s2->intraWidth - s1->intraWidth))) return d;
   if ((d = (s1->intraNhops - s2->intraNhops))) return d;
   return s1->startIndex - s2->startIndex;
}

static int cmpIntraScores(struct ncclGpuScore* scores, int count) {
  int intraWidth = scores[0].intraWidth;
  int intraNhops = scores[0].intraNhops;
  for (int i=1; i<count; i++) {
    if (scores[i].intraWidth != intraWidth || scores[i].intraNhops != intraNhops) return 1;
  }
  return 0;
}

static ncclResult_t getGpuIndex(struct ncclTopoSystem* system, int rank, int* index) {
  for (int g=0; g<system->nodes[GPU].count; g++) {
    if (system->nodes[GPU].nodes[g].gpu.rank == rank) {
      *index = g;
      return ncclSuccess;
    }
  }
  WARN("Could not find gpu rank %d\n", rank);
  return ncclInternalError;
}

static ncclResult_t getNetIndex(struct ncclTopoSystem* system, int64_t id, int* index) {
  for (int n=0; n<system->nodes[NET].count; n++) {
    if (system->nodes[NET].nodes[n].id == id) {
      *index = n;
      return ncclSuccess;
    }
  }
  WARN("Could not find net id %lx\n", id);
  return ncclInternalError;
}

static ncclResult_t getNetPaths(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoLinkList** netPaths) {
  int netId = graph->inter[graph->nChannels*2];
  int n;
  NCCLCHECK(getNetIndex(system, netId, &n));
  *netPaths=system->nodes[NET].nodes[n].paths[GPU];
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchNextGpuSort(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoNode* gpu, int* next, int* countPtr, int sortNet) {
  const uint64_t flag = 1ULL<<(graph->nChannels);
  int ngpus = system->nodes[GPU].count;
  struct ncclTopoLinkList* paths = gpu->paths[GPU];
  struct ncclTopoLinkList* netPaths = NULL;
  if (sortNet) NCCLCHECK(getNetPaths(system, graph, &netPaths));

  struct ncclGpuScore scores[NCCL_TOPO_MAX_NODES];
  memset(scores, 0, ngpus*sizeof(struct ncclGpuScore));
  int start = gpu-system->nodes[GPU].nodes;
  int count = 0;
  for (int i=1; i<ngpus; i++) {
    int g = (start+i)%ngpus;
    if (paths[g].count == 0) continue; // There is no path to that GPU
    if (system->nodes[GPU].nodes[g].used & flag) continue;
    scores[count].g = g;
    scores[count].startIndex = i;
    scores[count].intraNhops = paths[g].count;
    scores[count].intraWidth = paths[g].width;
    if (netPaths) {
      scores[count].interNhops = netPaths[g].count;
      scores[count].interPciWidth = gpuPciWidth(system->nodes[GPU].nodes+g);
      scores[count].interWidth = netPaths[g].width;
    }
    count++;
  }

  // Sort GPUs
  qsort(scores, count, sizeof(struct ncclGpuScore), cmpScore);

  // Check if all have the same intra-node score in which case we go reverse for sortNet = -1
  if (sortNet == -1 && cmpIntraScores(scores, count) == 0) {
    for (int i=0; i<count; i++) next[i] = scores[count-1-i].g;
  } else {
    for (int i=0; i<count; i++) next[i] = scores[i].g;
  }
  *countPtr = count;
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time);

// Try to keep all searchs within one second
#define NCCL_SEARCH_GLOBAL_TIMEOUT (3ULL<<19)
#define NCCL_SEARCH_TIMEOUT (1<<18)
#define NCCL_SEARCH_TIMEOUT_TREE (1<<17)
#define NCCL_SEARCH_TIMEOUT_SAMECHANNELS (1<<10)

#define FORCED_ORDER_PCI 1
#define FORCED_ORDER_REPLAY 2

ncclResult_t ncclTopoReplayGetGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int step, int* g) {
  *g = -1;
  if (graph->nChannels == 0) return ncclInternalError;
  int ngpus = system->nodes[GPU].count;
  int nextRank = graph->intra[(graph->nChannels-1)*ngpus+step+1];
  for (int i=0; i<ngpus; i++) if (system->nodes[GPU].nodes[i].gpu.rank == nextRank) {
    *g = i;
    return ncclSuccess;
  }
  if (*g == -1) return ncclInternalError;
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time);

ncclResult_t ncclTopoSearchTryGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time, int type, int index, int g) {
  const uint64_t flag = 1ULL<<(graph->nChannels);
  struct ncclTopoNode* gpu;
  NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, 1, &gpu));
  if (gpu) {
    gpu->used ^= flag;
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time));
    gpu->used ^= flag;
    NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, -1, &gpu));
  }
  return ncclSuccess;
}

static int ncclTopoCountXGMI(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  int count = 0;
  for (int c=0; c<graph->nChannels; c++) {
    for (int i=0; i<ngpus; i++) {
      int g = graph->intra[ngpus*c+i];
      int n = graph->intra[ngpus*c+((i+1)%ngpus)];
      struct ncclTopoNode *node;
      int j;
      for (j=0; j<ngpus; j++)
        if (system->nodes[GPU].nodes[j].gpu.rank == g) break;
      if (j<ngpus) {
        node = system->nodes[GPU].nodes+j;
        for (int k = 0; k<system->nodes[GPU].count; k++) {
          if (node->paths[GPU][k].count == 1) {
            struct ncclTopoLink* link = node->paths[GPU][k].list[0];
            struct ncclTopoNode* remNode = link->remNode;
            if (remNode->gpu.rank == n) {
              if (link->type == LINK_NVL)
                count ++;
            }
          }
        }
      }
    }
  }
  return count;
}

ncclResult_t ncclTopoCompareGraphs(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* refGraph, int* copy) {
  // 1. Constraint to get the same nChannels between Rings and Trees
  if (graph->nChannels < graph->minChannels) return ncclSuccess;

  // 2. Try to get better bandwidth
  if (graph->nChannels*graph->speedIntra < refGraph->nChannels*refGraph->speedIntra) return ncclSuccess;
  if (graph->nChannels*graph->speedIntra > refGraph->nChannels*refGraph->speedIntra) {
    *copy = 1;
    return ncclSuccess;
  }
  // 3. Less hops (but not at the price of going cross NICs)
  if (graph->crossNic == refGraph->crossNic && graph->nHops < refGraph->nHops) *copy = 1;

  // 4. Prefer graph with more XGMI connections
  if (graph->nChannels == refGraph->nChannels
    && ncclTopoCountXGMI(system, refGraph) < ncclTopoCountXGMI(system, graph)) *copy = 1;
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time) {
  if ((*time) <= 0) return ncclSuccess;
  (*time)--;

  int ngpus = system->nodes[GPU].count;
  if (step == ngpus) {
    // Determine whether we found a better solution or not
    int copy = 0;
    graph->nChannels++;
    NCCLCHECK(ncclTopoCompareGraphs(system, graph, saveGraph, &copy));
    if (copy) {
      memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
      if (graph->nChannels == graph->maxChannels) *time = -1;
    }
    if (graph->nChannels < graph->maxChannels) {
      NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
    }
    graph->nChannels--;
    return ncclSuccess;
  }
  graph->intra[graph->nChannels*ngpus+step] = gpu->gpu.rank;
  int g = gpu - system->nodes[GPU].nodes;
  if (step == backToNet) {
    // first get back to NIC
    if (system->nodes[NET].count) {
      int startNetIndex;
      NCCLCHECK(getNetIndex(system, graph->inter[graph->nChannels*2], &startNetIndex));
      struct ncclTopoNode* startNet = system->nodes[NET].nodes+startNetIndex;
      for (int n=0; n<system->nodes[NET].count; n++) {
        struct ncclTopoNode* net = system->nodes[NET].nodes+n;
        if (graph->pattern == NCCL_TOPO_PATTERN_TREE && net->id != startNet->id) continue; // Trees are symmetric
        if (graph->crossNic != 1 && (net->net.asic != startNet->net.asic || net->net.port != startNet->net.port)) continue;
        NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1, &net));
        if (net) {
          graph->inter[graph->nChannels*2+1] = net->id;
          NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, -1, backToFirstRank, forcedOrder, time));
          NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1, &net));
        }
      }
    }
  } else if (step < system->nodes[GPU].count-1) {
    // Go to next GPU
    int next[NCCL_TOPO_MAX_NODES];
    int count;
    if (forcedOrder == FORCED_ORDER_PCI) { // Try the PCI order
      next[0] = step+1;
      count = 1;
    } else if (forcedOrder == FORCED_ORDER_REPLAY) { // Try last channel order
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, step, next));
      count = 1;
    } else { // Normal search
      NCCLCHECK(ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count, backToNet == -1 ? 0 : backToNet == step+1 ? 1 : -1 ));
    }
    for (int i=0; i<count; i++) {
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, step+1, backToNet, backToFirstRank, forcedOrder, time, GPU, g, next[i]));
    }
  } else if (step == backToFirstRank) {
    // Find first GPU and loop back to it
    int p;
    NCCLCHECK(getGpuIndex(system, graph->intra[graph->nChannels*ngpus], &p));
    struct ncclTopoNode* firstGpu;
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, 1, &firstGpu));
    if (firstGpu) {
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step+1, backToNet, -1, forcedOrder, time));
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, -1, &firstGpu));
    }
  } else {
    // Next path
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int backToNet, int backToFirstRank, int* time) {
  const int speed = graph->speedInter;
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoNode* net = system->nodes[NET].nodes+n;
    struct ncclTopoNode* gpu;
    if (graph->collNet && net->net.collSupport == 0) continue;
    if (net->net.width < speed) continue;
    if (net->net.maxChannels == 0) continue;

    graph->inter[graph->nChannels*2] = net->id;
    for (int i=0; i<system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.width -= speed;
      }
    }
    net->net.maxChannels--;

    // First try to replay the last channel
    if (graph->nChannels > 0) {
      int g;
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, NET, n, g));
    }
    if (graph->nChannels == 0 || graph->sameChannels == 0) {
      if (graph->nChannels == 0) {
        // Always try the PCI order first to set a reference, but don't count in the timeout nor let it run for long
        struct ncclTopoLinkList* paths = net->paths[GPU];
        // find the first GPU that is closest to NIC
        int f = 0;
        for (int i = 0; i<system->nodes[GPU].count; i++)
          if (paths[i].count < paths[f].count) f = i;
        int t = 1 << 10;
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, &t, NET, n, f));
        if (t == -1) *time = -1;
      }

      // Then try the most local GPUs
      float maxWidth = 0;
      int minHops = 0xfffffff;
      struct ncclTopoLinkList* paths = net->paths[GPU];
      for (int g=0; g<system->nodes[GPU].count; g++) {
        if (paths[g].width > maxWidth) {
          maxWidth = paths[g].width;
          minHops = paths[g].count;
        } else if (paths[g].width == maxWidth && paths[g].count < minHops) {
          minHops = paths[g].count;
        }
      }
      if (maxWidth >= speed) {
        // In the first loop, avoid using GPUs in both directions between channels (one channel
        // sending from that GPU and one channel receiving to that GPU), since that usually leads
        // to lower BW.
        for (int tryGpuBidir=0; tryGpuBidir<2; tryGpuBidir++) {
          for (int g=0; g<system->nodes[GPU].count; g++) {
            if (paths[g].width == maxWidth && paths[g].count == minHops) {
              gpu = system->nodes[GPU].nodes+g;
              int gpuUsed = gpuPciWidth(gpu) > 0 ? 0 : 1;
              if (tryGpuBidir == gpuUsed) {
                NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, g));
              }
            }
          }
        }
      }
    }

    net->net.maxChannels++;
    for (int i=0; i<system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.width += speed;
      }
    }
  }
  return ncclSuccess;
}

/* Search Patterns
 *
 *     Intra-node
 * Ring            : GPU a -> GPU b -> .. -> GPU x -> GPU a
 * (=Split Tree Loop)
 * Tree            : GPU a -> GPU b -> .. -> GPU x
 * (=Split Tree)
 *
 *     Inter-node
 * Ring            : NET n -> GPU a -> GPU b -> .. -> GPU x -> NET n (or m if crossNic)
 * Tree            : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                              `--> NET n (or m if crossNic)
 * Split Tree      : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                                       `--> NET n (or m if crossNic)
 * Split Tree Loop : NET n -> GPU a -> GPU b -> .. -> GPU x -> GPU a
 *                                       `--> NET n (or m if crossNic)
 */
ncclResult_t ncclTopoSearchParams(struct ncclTopoSystem* system, int pattern, int* backToNet, int* backToFirstRank) {
  if (system->nodes[NET].count) {
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToNet = system->nodes[GPU].count-1;
    else if (pattern == NCCL_TOPO_PATTERN_TREE) *backToNet = 0;
    else *backToNet = 1;
    if (pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) *backToFirstRank = system->nodes[GPU].count-1;
    else *backToFirstRank = -1;
  } else {
    *backToNet = -1;
    if (pattern == NCCL_TOPO_PATTERN_RING || pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) *backToFirstRank = system->nodes[GPU].count-1;
    else *backToFirstRank = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time) {
  int backToNet, backToFirstRank;
  NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));
  if (system->nodes[NET].count) {
    // Start from NET
    ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
  } else {
    // Intra-node only.
    if (graph->nChannels == 0) {
      // Try PCI order first
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, -1, -1, 0));
    } else {
      // Also try to replay previous channel
      int g;
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, -1, -1, g));
    }
    if (graph->sameChannels == 0 || graph->nChannels == 0) {
      // Finally, try all other possibilities unless we are forced to use the same channels
      for (int g=0; g<system->nodes[GPU].count; g++) {
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, g));
      }
    }
  }
  return ncclSuccess;
}

/************************************/
/* User defined graph from XML file */
/************************************/

struct kvDict kvDictLinkType[] = { { "SYS", PATH_SYS }, { "PHB", PATH_PHB }, { "PIX", PATH_PIX }, { "PXB", PATH_PXB }, { "NVL", PATH_NVL }, { "LOC", PATH_LOC }, { NULL, 0 } };
ncclResult_t ncclTopoGetChannelFromXml(struct ncclXmlNode *xmlChannel, int c, struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  int* inter = graph->inter+2*c;
  int* intra = graph->intra+ngpus*c;
  int n=0, g=0;
  for (int s=0; s<xmlChannel->nSubs; s++) {
    struct ncclXmlNode* sub = xmlChannel->subs[s];
    int dev;
    NCCLCHECK(xmlGetAttrInt(sub, "dev", &dev));
    if (strcmp(sub->name, "net") == 0) {
      inter[n++] = dev;
    } else if (strcmp(sub->name, "gpu") == 0) {
      int rank = -1;
      for (int g=0; g<ngpus; g++) {
        if (system->nodes[GPU].nodes[g].gpu.dev == dev) rank = system->nodes[GPU].nodes[g].gpu.rank;
      }
      if (rank == -1) {
        WARN("XML Import Channel : dev %d not found.", dev);
        return ncclSystemError;
      }
      intra[g++] = rank;
    }
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetGraphFromXmlSub(struct ncclXmlNode *xmlGraph, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels) {
  int id;
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "id", &id));
  if (graph->id != id) return ncclSuccess;

  int crossNic;
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "crossnic", &crossNic));
  if (graph->crossNic == 0 && crossNic == 1) return ncclSuccess;
  graph->crossNic = crossNic;

  NCCLCHECK(xmlGetAttrInt(xmlGraph, "pattern", &graph->pattern));
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "nchannels", &graph->nChannels));
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedintra", &graph->speedIntra));
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedinter", &graph->speedInter));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeintra", &str));
  NCCLCHECK(kvConvertToInt(str, &graph->typeIntra, kvDictLinkType));
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeinter", &str));
  NCCLCHECK(kvConvertToInt(str, &graph->typeInter, kvDictLinkType));
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "samechannels", &graph->sameChannels));
  for (int s=0; s<xmlGraph->nSubs; s++) {
    NCCLCHECK(ncclTopoGetChannelFromXml(xmlGraph->subs[s], s, system, graph));
  }
  *nChannels = xmlGraph->nSubs;
  return ncclSuccess;
}
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode *xmlGraphs, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels) {
  for (int s=0; s<xmlGraphs->nSubs; s++) {
    NCCLCHECK(ncclTopoGetGraphFromXmlSub(xmlGraphs->subs[s], system, graph, nChannels));
  }
  return ncclSuccess;
}

/* And the reverse : graph->xml */
ncclResult_t ncclTopoGetXmlFromChannel(struct ncclTopoGraph* graph, int c, struct ncclTopoSystem* system, struct ncclXml *xml, struct ncclXmlNode* parent) {
  struct ncclXmlNode* xmlChannel;
  int ngpus = system->nodes[GPU].count;
  int* inter = graph->inter+2*c;
  int* intra = graph->intra+ngpus*c;
  NCCLCHECK(xmlAddNode(xml, parent, "channel", &xmlChannel));
  struct ncclXmlNode* node;
  if (system->nodes[NET].count) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));
    NCCLCHECK(xmlSetAttrInt(node, "dev", inter[0]));
  }
  for (int g=0; g<ngpus; g++) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "gpu", &node));
    int dev = -1;
    for (int i=0; i<ngpus; i++) {
      if (system->nodes[GPU].nodes[i].gpu.rank == intra[g]) dev = system->nodes[GPU].nodes[i].gpu.dev;
    }
    if (dev == -1) {
      WARN("XML Export Channel : rank %d not found.", intra[g]);
      return ncclInternalError;
    }
    NCCLCHECK(xmlSetAttrInt(node, "dev", dev));
  }
  if (system->nodes[NET].count) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));
    NCCLCHECK(xmlSetAttrInt(node, "dev", inter[1]));
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetXmlFromGraph(struct ncclTopoGraph* graph, struct ncclTopoSystem* system, struct ncclXml *xml, struct ncclXmlNode* parent) {
  struct ncclXmlNode* xmlGraph;
  NCCLCHECK(xmlAddNode(xml, parent, "graph", &xmlGraph));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "id", graph->id));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "pattern", graph->pattern));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "crossnic", graph->crossNic));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "nchannels", graph->nChannels));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedintra", graph->speedIntra));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedinter", graph->speedInter));
  const char* str;
  NCCLCHECK(kvConvertToStr(graph->typeIntra, &str, kvDictLinkType));
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeintra", str));
  NCCLCHECK(kvConvertToStr(graph->typeInter, &str, kvDictLinkType));
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeinter", str));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "samechannels", graph->sameChannels));
  for (int c=0; c<graph->nChannels; c++) {
    NCCLCHECK(ncclTopoGetXmlFromChannel(graph, c, system, xml, xmlGraph));
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system, struct ncclXml *xml) {
  xml->maxIndex = 0;
  struct ncclXmlNode* xmlGraphs;
  NCCLCHECK(xmlAddNode(xml, NULL, "graphs", &xmlGraphs));
  NCCLCHECK(xmlSetAttrInt(xmlGraphs, "version", NCCL_GRAPH_XML_VERSION));
  for (int g=0; g<ngraphs; g++) {
    NCCLCHECK(ncclTopoGetXmlFromGraph(graphs[g], system, xml, xmlGraphs));
  }
  return ncclSuccess;
}

/* Parse user defined rings. Format is like :
 * "0 1|1 0|0 1 2 3|3 2 1 0|0 2 3 1|1 3 2 0|0 1 2 3 4 5 6 7|7 6 5 4 3 2 1 0"
 * Rings with a non-matching number of ranks are ignored so we can provide
 * rings for multiple cases.
 */
#define MAX_ENV_RANKS 512
static ncclResult_t parseGraph(const char* str, int* nChannelsRet, int ngpus, int* channels) {
  int ranks[MAX_ENV_RANKS];
  int nChannels = 0;
  int rank = 0;
  int offset = 0;
  int status = 0; // 0 : between numbers, 1 : inside number
  do {
    int digit = str[offset] - '0';
    if (digit >= 0 && digit <= 9) {
      if (status == 0) {
        ranks[rank] = digit;
        status = 1;
      } else {
        ranks[rank] = ranks[rank]*10+digit;
      }
    } else {
      if (status == 1) {
        rank++;
        if (rank == MAX_ENV_RANKS) goto end;
      }
      status = 0;
      if (str[offset] == '|' || str[offset] == '\0') {
        // Ignore if ngpus doesn't match
        if (rank != ngpus) goto newchannel;

        for (int r=0; r<ngpus; r++) {
          int rank = ranks[r];
          // Ignore if ranks are out of bounds
          if (rank < 0 || rank >= ngpus) goto newchannel;
          // Ignore if ranks are duplicate
          for (int i=0; i<r; i++)
            if (ranks[i] == rank) goto newchannel;

          channels[nChannels*ngpus+r] = rank;
        }
        nChannels++;
newchannel:
        rank = 0;
      }
    }
  } while (str[offset++] != 0);
end:
  *nChannelsRet = nChannels;
  return ncclSuccess;
}

static ncclResult_t parseChordalRing(struct ncclTopoSystem* system, char **str) {
  static const char *ringBase = "0 1 2 3 5 4 7 6|0 2 4 1 7 3 6 5|0 3 1 5 7 2 6 4|0 6 7 4 5 3 2 1|0 5 6 3 7 1 4 2|0 4 6 2 7 5 1 3";
  static char ringRemap[256];
  int id[8], dist[8];
  int i;

  *str = 0;
  int ngpus = system->nodes[GPU].count;
  if (ngpus != 8)
    return ncclSuccess;
  // validate chordal ring and calculate distance
  for (i=0; i<ngpus; i++) {
    struct ncclTopoNode* node = system->nodes[GPU].nodes+i;
    if (node->paths[GPU] == NULL) continue;
    int sum = ngpus*(ngpus-1)/2 - node->gpu.dev;
    int count = 0;
    for (int n = 0; n<ngpus; n++) {
      struct ncclTopoLink* link;
      for (link = node->links; link->remNode; link++) {
        if (link->remNode->gpu.dev == n) break;
      }
      if (!link->remNode) continue;
      if (link->type != LINK_NVL) continue;
      sum -= system->nodes[GPU].nodes[n].gpu.dev;
      count ++;
    }
    if(count != ngpus-2 || sum < 0 || sum > ngpus-1) {
      return ncclSuccess;
    }
    dist[i] = sum;
  }
  // remap GPU ids
  for (i = 0; i<ngpus; i++) id[i] = i;
  for (i = 0; i<ngpus; i++) {
    if (dist[i] == ngpus-1-i) continue;
    int j, m, n, temp;
    for (j=i+1; j < ngpus; j++)
      if(dist[j] == ngpus-1-i) break;
    m = dist[i]; n = dist[j]; dist[i] = n; dist[j] = m;
    temp = id[m]; id[m] = id[n]; id[n] = temp; temp =dist[m];
    dist[m] = dist[n]; dist[n] = temp;
  }
  // create chordal ring based on reference and remapped ids
  for (i = 0; i <strlen(ringBase); i++) {
    if (ringBase[i] >= '0' && ringBase[i] <= '9')
      ringRemap[i] = id[ringBase[i]-'0']+'0';
    else
      ringRemap[i] = ringBase[i];
  }
  ringRemap[i] = 0;
  *str = ringRemap;
  system->type = RCCL_TOPO_CR8G;
  INFO(NCCL_GRAPH, "Use chordal ring: %s", ringRemap);
  return ncclSuccess;
}

static bool getGpuNetCount(struct ncclTopoSystem* system, int id, int *ngpu, int *nnet) {
  *ngpu = 0; *nnet = 0;
  int i;
  if (ncclTopoIdToIndex(system, CPU, id, &i) == ncclInternalError) return false;
  for (int n = 0; n < system->nodes[NET].count; n++)
    if (system->nodes[NET].nodes[n].paths[CPU][i].count == 2) (*nnet)++;
  for (int n = 0; n < system->nodes[GPU].count; n++)
    if (system->nodes[GPU].nodes[n].paths[CPU][i].count == 2) (*ngpu)++;
  return true;
}

static bool findGpuByXGMI(struct ncclTopoSystem* system, int cpu1, int cpu2, int *gpu1, int *gpu2, int ex1, int ex2) {
  int n, m, idx, gid;
  int ngpus = system->nodes[GPU].count;
  *gpu1 = -1; *gpu2 = -1;
  int c1, c2;
  if (ncclTopoIdToIndex(system, CPU, cpu1, &c1) == ncclInternalError) return false;
  if (ncclTopoIdToIndex(system, CPU, cpu2, &c2) == ncclInternalError) return false;
  for (n = 0; n < ngpus; n++) {
    if (system->nodes[GPU].nodes[n].gpu.dev == ex1) continue;
    if (system->nodes[GPU].nodes[n].paths[CPU][c1].count != 2) continue;
    struct ncclTopoNode* node = system->nodes[GPU].nodes+n;
    if (node->paths[GPU] == NULL) continue;
    idx = -1; gid = 0;
    for (m = 0; m < ngpus; m++) {
      if (system->nodes[GPU].nodes[m].gpu.dev == ex2) continue;
      if (system->nodes[GPU].nodes[m].paths[CPU][c2].count != 2) continue;
      struct ncclTopoLink* link;
      for (link = node->links; link->remNode; link++) {
        if (link->remNode->gpu.dev == system->nodes[GPU].nodes[m].gpu.dev) break;
      }
      if (!link->remNode) continue;
      if (link->type == LINK_NVL) {
        if (idx == -1 || (idx != -1 && system->nodes[GPU].nodes[m].id < gid)) {
          idx = m;
          gid = system->nodes[GPU].nodes[m].id;
        }
      }
    }
    if (idx != -1) break;
  }
  if (n < ngpus) {
    *gpu1 = system->nodes[GPU].nodes[n].gpu.dev;
    *gpu2 = system->nodes[GPU].nodes[idx].gpu.dev;
    return true;
  }
  return false;
}

static bool validate4P1H(struct ncclTopoSystem* system, int *hive) {
  int g, n, m;
  int ngpus = system->nodes[GPU].count;
  for (g = 0; g < 4; g++) {
    int gpu = hive[g];
    int next_gpu = hive[(g+1)%4];
    for (n = 0; n < ngpus; n++) {
      if (system->nodes[GPU].nodes[n].gpu.dev != gpu) continue;
      struct ncclTopoNode* node = system->nodes[GPU].nodes+n;
      if (node->paths[GPU] == NULL) continue;
      for (m = 0; m < ngpus; m++) {
        struct ncclTopoLink* link;
        for (link = node->links; link->remNode; link++) {
          if (link->remNode->gpu.dev == next_gpu) break;
        }
        if (!link->remNode) continue;
        if (link->type == LINK_NVL) break;
      }
      if (m < ngpus) break;
    }
    if (n < ngpus) continue;
    else break;
  }
  if (g < 4) return false;
  else return true;
}

static ncclResult_t parseRome4P2H(struct ncclTopoSystem* system, char **str) {
  static const char *ringBase = "6 7 4 5 1 0 3 2|7 6 2 3 0 1 5 4";
  static char ringRemap[64];
  int id[8], dist[8];
  int i;

  *str = 0;
  int ngpus = system->nodes[GPU].count;
  int ncpus = system->nodes[CPU].count;
  // 8 GPUs and 4 numa nodes only
  if (ngpus != 8 || ncpus != 4)
    return ncclSuccess;
  // only valid on Rome
  int arch, vendor, model;
  NCCLCHECK(ncclTopoCpuType(system, &arch, &vendor, &model));
  if (arch != NCCL_TOPO_CPU_ARCH_X86 || vendor != NCCL_TOPO_CPU_VENDOR_AMD || model != NCCL_TOPO_CPU_TYPE_ROME)
    return ncclSuccess;
  // number of GPUs and NICs on each numa node is used as first screening pattern
  char pattern[9];
  for (int i = 0; i < ncpus; i++) {
    int g, n;
    if (!getGpuNetCount(system, i, &g, &n)) return ncclSuccess;
    pattern[i*2] = '0' + g;
    pattern[i*2+1] = '0' + n;
  }
  pattern[8] = 0;
  if (strcmp(pattern, "10302120")) return ncclSuccess;
  // identify GPUs for pattern "10302120"
  int g[8];
  if (!findGpuByXGMI(system, 1, 3, &g[2], &g[6], -1, -1)) return ncclSuccess;
  if (!findGpuByXGMI(system, 2, 3, &g[4], &g[7], -1, -1)) return ncclSuccess;
  if (!findGpuByXGMI(system, 0, 1, &g[0], &g[1], -1, -1)) return ncclSuccess;
  if (!findGpuByXGMI(system, 1, 2, &g[3], &g[5], g[1], g[4])) return ncclSuccess;
  // finally verify two XGMI hives for pattern "10302120"
  int h1[4], h2[4];
  h1[0] = g[0]; h1[1] = g[1]; h1[2] = g[5]; h1[3] = g[3];
  h2[0] = g[2]; h2[1] = g[4]; h2[2] = g[7]; h2[3] = g[6];
  if (!validate4P1H(system, h1)) return ncclSuccess;
  if (!validate4P1H(system, h2)) return ncclSuccess;
  // passed all validation
  // create 4P2H based on reference and remapped ids
  for (i = 0; i <strlen(ringBase); i++) {
    if (ringBase[i] >= '0' && ringBase[i] <= '9')
      ringRemap[i] = g[ringBase[i]-'0'] + '0';
    else
      ringRemap[i] = ringBase[i];
  }
  ringRemap[i] = 0;
  *str = ringRemap;
  system->type = RCCL_TOPO_4P2H_ROME;
  INFO(NCCL_GRAPH, "Use 4P2H on Rome: %s", ringRemap);
  return ncclSuccess;
}

#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
float speedArray[] = { 24.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.4, 1.2, 0.24, 0.12 };
#else
float speedArray[] = { 42.0, 24.0, 21.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.4, 1.2, 0.24, 0.12 };
#endif
#define NSPEEDS (sizeof(speedArray)/sizeof(float))

ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  int crossNic = (system->nodes[NET].count > 1) && graph->crossNic ? 1 : 0;
  graph->speedIntra = graph->speedInter = 0;
  if (graph->crossNic == 2) graph->crossNic = 0;
  graph->typeIntra = ngpus == 1 ? PATH_LOC : PATH_NVL;
  graph->typeInter = PATH_PIX;
  graph->nChannels = 0;
  graph->sameChannels = 1;

  char* str = getenv("NCCL_GRAPH_FILE");
  if (str) {
    INFO(NCCL_ENV, "NCCL_GRAPH_FILE set by environment to %s", str);
    struct ncclXml* xml;
    NCCLCHECK(ncclCalloc(&xml, 1));
    NCCLCHECK(ncclTopoGetXmlGraphFromFile(str, xml));
    int nChannels;
    NCCLCHECK(ncclTopoGetGraphFromXml(xml->nodes, system, graph, &nChannels));
    INFO(NCCL_GRAPH, "Search %d : %d channels loaded from XML graph", graph->id, nChannels);
    free(xml);
    if (graph->nChannels > 0) return ncclSuccess;
  }

  if (!str) NCCLCHECK(parseChordalRing(system, &str));
  if (!str) NCCLCHECK(parseRome4P2H(system, &str));
  if (str) {
    NCCLCHECK(parseGraph(str, &graph->nChannels, ngpus, graph->intra));
    for (int i=0; i<graph->nChannels*ngpus; i++) {
      // Translate gpu numbers into ranks
      int j = 0;
      for (j = 0; j < system->nodes[GPU].count; j++)
        if (graph->intra[i] == system->nodes[GPU].nodes[j].gpu.dev)
          break;
      if (j < system->nodes[GPU].count)
        graph->intra[i] = system->nodes[GPU].nodes[j].gpu.rank;
      else
        return ncclInternalError;
    }
    graph->speedIntra = graph->speedInter = system->maxWidth;
    if (system->nodes[NET].count) {
      // do not change ring order for multi node 4P2H on Rome
      if (system->type == RCCL_TOPO_4P2H_ROME) {
        for (int n = 0; n < graph->nChannels; n++)
          graph->inter[n*2] = graph->inter[n*2+1] = n%system->nodes[NET].count;
      } else {
        int *intra, *used;
        graph->nChannels = system->nodes[NET].count;
        NCCLCHECK(ncclCalloc(&intra, ngpus));
        NCCLCHECK(ncclCalloc(&used,system->nodes[NET].count));
        for (int n = 0; n < system->nodes[NET].count; n++) {
          graph->inter[n*2] = graph->inter[n*2+1] = n;
          struct ncclTopoNode* net = system->nodes[NET].nodes+n;
          struct ncclTopoLinkList* paths = net->paths[GPU];
          // find the first unsed GPU that is closest to NIC
          int f, m;
          for (f = 0; f < ngpus; f++) {
            int j = 0; for (j = 0; j < n; j++) if(used[j] == system->nodes[GPU].nodes[f].gpu.rank) break;
            if(j >= n) break;
          }
          for (int i = 0; i < ngpus; i++) {
            int j = 0; for (j = 0; j < n; j++) if(used[j] == system->nodes[GPU].nodes[i].gpu.rank) break;
            if (j < n) continue;
            if (paths[i].count < paths[f].count) f = i;
          }
          for (m = 0; m<ngpus; m++) if (graph->intra[n*ngpus+m] == system->nodes[GPU].nodes[f].gpu.rank) break;
          used[n] = graph->intra[n*ngpus+m];
          for (int i = 0; i < ngpus; i++) intra[i] = graph->intra[n*ngpus+((i+m)%ngpus)];
          for (int i = 0; i < ngpus; i++) graph->intra[n*ngpus+i] = intra[i];
        }
        free(used);
        free(intra);
      }
    }
    if (graph->nChannels) return ncclSuccess;
  }

  if (ngpus == 1) if (graph->pattern != NCCL_TOPO_PATTERN_RING) graph->pattern = NCCL_TOPO_PATTERN_TREE;

  struct ncclTopoGraph tmpGraph;
  memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));

  // First try crossnic, then decrease speed and finally increase speedIntra.
  tmpGraph.pattern = graph->pattern;
  int pass = 1;
  int speedIndex = 0;
  while (speedArray[speedIndex] > system->maxWidth && speedIndex < NSPEEDS-1) speedIndex++;
  tmpGraph.speedIntra = tmpGraph.speedInter = speedArray[speedIndex];
  int64_t globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;

search:
  int time = tmpGraph.sameChannels ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS :
    tmpGraph.pattern == NCCL_TOPO_PATTERN_TREE ? NCCL_SEARCH_TIMEOUT_TREE : NCCL_SEARCH_TIMEOUT;
  tmpGraph.nChannels = 0;
  globalTimeout -= time;

  NCCLCHECK(ncclTopoSearchRec(system, &tmpGraph, graph, &time));
#if 0
  printf("Pattern %d, crossNic %d, Speed %g/%g, type %d/%d, channels %d-%d sameChannels %d -> nChannels %dx%g/%g %s\n", tmpGraph.pattern, tmpGraph.crossNic, tmpGraph.speedInter, tmpGraph.speedIntra, tmpGraph.typeInter, tmpGraph.typeIntra, tmpGraph.minChannels, tmpGraph.maxChannels, tmpGraph.sameChannels, graph->nChannels, graph->speedInter, graph->speedIntra, time == 0 ? "TIMEOUT" : "");
  for (int c=0; c<graph->nChannels; c++) {
    printf("%2d : ", c);
    for (int g=0; g<ngpus; g++) {
      printf("%d ", graph->intra[c*ngpus+g]);
    }
    printf("\n");
  }
#endif
  // Optimal solution, stop here
  if (graph->nChannels == graph->maxChannels && graph->speedInter == system->maxWidth) goto done;

  if (pass == 1) {
    // First pass, we don't have a solution yet ; try other options

    // Try having different channels
    if (tmpGraph.sameChannels == 1) {
      tmpGraph.sameChannels = 0;
      goto search;
    }
    tmpGraph.sameChannels = 1;

    if (time != -1) globalTimeout += time;
    else globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;
    if (globalTimeout < 0) goto done;

    int maxTypeIntra = system->nodes[NET].count > 0 ? tmpGraph.typeInter : PATH_SYS;
    if (tmpGraph.typeIntra < maxTypeIntra && (graph->nChannels == 0 || tmpGraph.typeIntra < graph->typeIntra)) {
      tmpGraph.typeIntra += 1;
      goto search;
    }
    tmpGraph.typeIntra = ngpus == 1 ? PATH_LOC : PATH_NVL;
    if (system->nodes[NET].count > 0 && tmpGraph.typeInter < PATH_SYS && (graph->nChannels == 0 || tmpGraph.typeInter < graph->typeInter || tmpGraph.typeInter < PATH_PXB)) {
      tmpGraph.typeInter += 1;
      goto search;
    }
    tmpGraph.typeInter = PATH_PIX;

    // Try a simpler tree
    if (tmpGraph.pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) {
      tmpGraph.pattern = NCCL_TOPO_PATTERN_SPLIT_TREE;
      goto search;
    }
    if (tmpGraph.pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) {
      tmpGraph.pattern = NCCL_TOPO_PATTERN_TREE;
      goto search;
    }
    tmpGraph.pattern = graph->pattern;

    if (crossNic && tmpGraph.crossNic == 0) {
      // Try again with crossNic if permitted
      tmpGraph.crossNic = crossNic;
      goto search;
    }
    tmpGraph.crossNic = 0;

    // Decrease speed until we find a solution
    if ((speedIndex < NSPEEDS-1) && (graph->nChannels == 0 || (speedArray[speedIndex+1]/graph->speedInter > .49))) {
      tmpGraph.speedInter = tmpGraph.speedIntra = speedArray[++speedIndex];
      goto search;
    }
    speedIndex = 0;
    while (speedArray[speedIndex] > system->maxWidth && speedIndex < NSPEEDS-1) speedIndex++;
    tmpGraph.speedIntra = tmpGraph.speedInter = speedArray[speedIndex];

  }

done:
  // We have a solution. Start from that solution and move to pass 2.
  if (pass == 1) {
    time = -1;
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
    speedIndex = 0;
    while (speedArray[speedIndex] > graph->speedInter && speedIndex < NSPEEDS-1) speedIndex++;
    tmpGraph.speedIntra = tmpGraph.speedInter = speedArray[speedIndex];
    tmpGraph.minChannels = graph->nChannels;
    pass = 2;
  }

  // 3. See if we can increase speedIntra for trees (2 nodes or collnet)
  if (pass == 2) {
    if (time != 0 && graph->pattern != NCCL_TOPO_PATTERN_RING &&
        tmpGraph.speedIntra == graph->speedIntra && tmpGraph.speedIntra < tmpGraph.speedInter*2 &&
        speedIndex > 0) {
      tmpGraph.speedIntra = speedArray[--speedIndex];
      goto search;
    }
    time = -1;
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
  }

  if (graph->nChannels == 0 && graph->collNet == 0) {
    WARN("Could not find a path for pattern %d, falling back to simple order\n", graph->pattern);
    for (int i=0; i<ngpus; i++) graph->intra[i] = system->nodes[GPU].nodes[i].gpu.rank;
    graph->inter[0] = graph->inter[1] = 0;
    graph->speedIntra = graph->speedInter = 0.1;
    graph->typeIntra = graph->typeInter = PATH_SYS;
    graph->nChannels = 1;
  }

  if (graph->speedIntra >= 25.0) {
    int dupChannels = std::min(graph->nChannels*2, graph->maxChannels);
    memcpy(graph->intra+graph->nChannels*ngpus, graph->intra, (dupChannels-graph->nChannels)*ngpus*sizeof(int));
    memcpy(graph->inter+graph->nChannels*2,graph->inter, (dupChannels-graph->nChannels)*2*sizeof(int));
    graph->speedIntra /= DIVUP(dupChannels, graph->nChannels);
    graph->speedInter /= DIVUP(dupChannels, graph->nChannels);
    graph->nChannels = dupChannels;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  INFO(NCCL_GRAPH, "Pattern %d, crossNic %d, nChannels %d, speed %f/%f, type %s/%s, sameChannels %d", graph->pattern, graph->crossNic, graph->nChannels, graph->speedIntra, graph->speedInter, topoPathTypeStr[graph->typeIntra], topoPathTypeStr[graph->typeInter], graph->sameChannels);
  int ngpus = system->nodes[GPU].count;

  char line[1024];
  for (int c=0; c<graph->nChannels; c++) {
    sprintf(line, "%2d :", c);
    int offset = strlen(line);
    if (system->nodes[NET].count > 0) {
      sprintf(line+offset, " %s/%d", topoNodeTypeStr[NET], graph->inter[2*c]);
      offset = strlen(line);
    }
    for (int i=0; i<ngpus; i++) {
      sprintf(line+offset, " %s/%d", topoNodeTypeStr[GPU], graph->intra[ngpus*c+i]);
      offset = strlen(line);
    }
    if (system->nodes[NET].count > 0) {
      sprintf(line+offset, " %s/%d", topoNodeTypeStr[NET], graph->inter[2*c+1]);
      offset = strlen(line);
    }
    INFO(NCCL_GRAPH, "%s", line);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs) {
  char* str = getenv("NCCL_GRAPH_DUMP_FILE");
  if (str) {
    INFO(NCCL_ENV, "NCCL_GRAPH_DUMP_FILE set by environment to %s", str);
    struct ncclXml* xml;
    NCCLCHECK(ncclCalloc(&xml, 1));
    NCCLCHECK(ncclTopoGetXmlFromGraphs(ngraphs, graphs, system, xml));
    NCCLCHECK(ncclTopoDumpXmlToFile(str, xml));
    free(xml);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNetDev(struct ncclTopoSystem* system, int rank, struct ncclTopoGraph* graph, int channelId, int* dev) {
  if (graph) {
    // Honor the net device in the graph
    int channel = channelId%graph->nChannels;
    int ngpus = system->nodes[GPU].count;
    int index = graph->intra[channel*ngpus] == rank ? 0 : 1;
    *dev = graph->inter[channel*2+index];
  } else {
    int64_t id;
    NCCLCHECK(ncclTopoGetLocalNet(system, rank, &id, channelId));
    *dev = id;
  }
  return ncclSuccess;
}
