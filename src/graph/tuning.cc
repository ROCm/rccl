/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "devcomm.h"
#include "comm.h"
#include "topo.h"

NCCL_PARAM(Nthreads, "NTHREADS", -2);
NCCL_PARAM(Ll128Nthreads, "LL128_NTHREADS", -2);

static int getNthreads(const char* name, int env, int min, int max, int def) {
  int nt = env;
  if (nt > 0) {
    if (nt % WARP_SIZE != 0) {
      WARN("Invalid %s %d (must be a multiple of %d)", name, nt, WARP_SIZE);
      nt = max;
    } else if (nt > max) {
      WARN("Invalid %s %d (maximum %d).", name, nt, max);
      nt = max;
    } else if (nt < min) {
      WARN("Invalid %s %d (minimum %d).", name, nt, min);
      nt = min;
     }
  } else {
    nt = def;
  }
  return nt;
}

ncclResult_t parseList(const char* str, const char* elems[], int nelems, int* list) {
  int def, set;
  if (str[0] == '^') {
    def = 1; set = 0; str++;
  } else {
    def = 0; set = 1;
  }
  for (int i=0; i<nelems; i++) list[i] = def;
  char* tokStr = strdup(str);
  char* tmpStr;
  char* token = strtok_r(tokStr, ",", &tmpStr);
  while (token) {
    for (int i=0; i<nelems; i++)
      if (strcasecmp(token, elems[i]) == 0) list[i] = set;
    token = strtok_r(NULL, ",", &tmpStr);
  }
  free(tokStr);
  return ncclSuccess;
}

// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
static const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = { { 12.0, 12.0, 17.0 }, { 12.0, 12.0, 17.0 }, { 12.0, 12.0, 17.0 } };

// NVLink, PCI, Network
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2

struct tuningModel {
  float hwLat [3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bwRatio [2][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][27];
  float ringCorrectionFactor[NCCL_NUM_PROTOCOLS][27];
};

static struct tuningModel tuning_model_0 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* Ring (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* CollNet (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNet (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 28.3, 28.3, 45.4 }, /* Ring (LL/LL128/Simple)*/ { 2.0, 2.0, 24.1 }, /* CollNet (LL/LL128/Simple)*/ { 28.3, 28.3, 45.4 } },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.06, 1.00, 1.30 }, /* Ring (LL/LL128/Simple)*/ { 0.07, 1.00, 1.00 }, /* CollNet (LL/LL128/Simple)*/ { 1.00, 1.00, 1.00 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.06, 1.00, 0.30 }, /* Ring (LL/LL128/Simple)*/ { 0.07, 1.00, 1.00 }, /* CollNet (LL/LL128/Simple)*/ { 1.00, 1.00, 1.00 } },
  },

  .treeCorrectionFactor = {
    { 0.3, 0.9, 0.8, 0.7, 0.6, 0.3, 0.1, 0.4, 0.8, 0.8, 0.5, 0.3, 0.4, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.3, 0.9, 0.8, 0.7, 0.6, 0.3, 0.1, 0.4, 0.8, 0.8, 0.5, 0.3, 0.4, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.4, 1.0, 1.0, 0.8, 1.0, 1.0, 0.3, 1.0, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
  },

  .ringCorrectionFactor = {
    { 0.2, 0.7, 0.7, 0.6, 0.6, 0.3, 0.2, 0.5, 1.0, 1.0, 0.8, 0.6, 0.8, 0.6, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.2, 0.7, 0.7, 0.6, 0.6, 0.3, 0.2, 0.5, 1.0, 1.0, 0.8, 0.6, 0.8, 0.6, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 1.0, 0.6, 0.9, 1.0, 0.7, 0.7, 1.0, 0.6, 0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },
};

static struct tuningModel tuning_model_1 {
  .hwLat =
  { /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* Ring (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* CollNet (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNet (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 33.0, 33.0, 15.8 }, /* Ring (LL/LL128/Simple)*/ { 5.1, 5.1, 68.8 }, /* CollNet (LL/LL128/Simple)*/ { 33.0, 33.0, 15.8 } },
  },

  .bwRatio =
  { /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.12, 1.00, 0.99 }, /* Ring (LL/LL128/Simple)*/ { 0.12, 1.00, 1.00 }, /* CollNet (LL/LL128/Simple)*/ { 1.00, 1.00, 1.00 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.15, 1.00, 0.42 }, /* Ring (LL/LL128/Simple)*/ { 0.20, 1.00, 1.00 }, /* CollNet (LL/LL128/Simple)*/ { 1.00, 1.00, 1.00 } },
  },

  .treeCorrectionFactor = {
    { 0.5, 0.4, 0.7, 0.6, 1.0, 1.0, 0.5, 0.4, 0.1, 0.5, 0.4, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, },
    { 0.5, 0.4, 0.7, 0.6, 1.0, 1.0, 0.5, 0.4, 0.1, 0.5, 0.4, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.4, 0.5, 0.1, 0.6, 1.0, 1.0, 1.0, 0.6, 0.5, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5, 0.3, 0.3, },
  },

  .ringCorrectionFactor = {
    { 1.0, 0.5, 1.0, 1.0, 0.6, 0.7, 1.0, 1.0, 0.2, 1.0, 0.9, 0.7, 1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.5, 0.3, 0.2, 0.1, 0.1, },
    { 1.0, 0.5, 1.0, 1.0, 0.6, 0.7, 1.0, 1.0, 0.2, 1.0, 0.9, 0.7, 1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.5, 0.3, 0.2, 0.1, 0.1, },
    { 0.3, 1.0, 0.3, 0.1, 0.1, 0.1, 0.3, 0.7, 1.0, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.9, 1.0, 1.0, 1.0, 1.0, },
  },
};

static struct tuningModel tuning_model_2 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* Ring (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* CollNet (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNet (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 27.9, 27.9, 15.8 }, /* Ring (LL/LL128/Simple)*/ { 12.1, 12.1, 68.8 }, /* CollNet (LL/LL128/Simple)*/ { 27.9, 27.9, 15.8 } },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.07, 1.00, 0.99 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 1.00, 1.00 }, /* CollNet (LL/LL128/Simple)*/ { 1.00, 1.00, 1.00 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.07, 1.00, 0.42 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 1.00, 1.00 }, /* CollNet (LL/LL128/Simple)*/ { 1.00, 1.00, 1.00 } },
  },

  .treeCorrectionFactor = {
    { 0.1, 0.4, 0.3, 0.3, 0.2, 0.4, 0.5, 0.1, 0.1, 0.6, 0.7, 0.7, 0.8, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.1, 0.4, 0.3, 0.3, 0.2, 0.4, 0.5, 0.1, 0.1, 0.6, 0.7, 0.7, 0.8, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.1, 0.6, 0.9, 0.8, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.9, 0.9, 1.0, 1.0, 1.0, },
  },

  .ringCorrectionFactor = {
    { 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.5, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },
};

static struct tuningModel tuning_model_3 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* Ring (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* CollNet (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNet (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 17.4, 17.4, 40.3 }, /* Ring (LL/LL128/Simple)*/ { 4.1, 4.1, 40.6 }, /* CollNet (LL/LL128/Simple)*/ { 17.4, 17.4, 40.3 } },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.08, 1.00, 0.95 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 1.00, 1.00 }, /* CollNet (LL/LL128/Simple)*/ { 1.00, 1.00, 1.00 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.08, 1.00, 0.41 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 1.00, 1.00 }, /* CollNet (LL/LL128/Simple)*/ { 1.00, 1.00, 1.00 } },
  },

  .treeCorrectionFactor = {
    { 0.6, 1.0, 1.0, 1.0, 0.1, 0.2, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.9, 0.8, 0.6, 0.4, 0.5, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.6, 1.0, 1.0, 1.0, 0.1, 0.2, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.9, 0.8, 0.6, 0.4, 0.5, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, },
  },

  .ringCorrectionFactor = {
    { 0.5, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 0.3, 0.7, 0.8, 0.5, 0.4, 0.2, 0.1, 0.1, 0.1, 0.3, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.5, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 0.3, 0.7, 0.8, 0.5, 0.4, 0.2, 0.1, 0.1, 0.1, 0.3, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },
};

static struct tuningModel tuning_model_4 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.8, 1.4, 2.5 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 2.2, 3.6 }, /* CollNet (LL/LL128/Simple)*/ { 0.8, 1.4, 2.5 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNet (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 32.2, 34.4, 47.6 }, /* Ring (LL/LL128/Simple)*/ { 35.4, 87.8, 209.2 }, /* CollNet (LL/LL128/Simple)*/ { 32.2, 34.4, 47.6 } },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.16, 1.09, 1.61 }, /* Ring (LL/LL128/Simple)*/ { 0.15, 0.41, 1.00 }, /* CollNet (LL/LL128/Simple)*/ { 1.00, 1.00, 1.00 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.16, 1.09, 1.08 }, /* Ring (LL/LL128/Simple)*/ { 0.15, 0.41, 1.00 }, /* CollNet (LL/LL128/Simple)*/ { 1.00, 1.00, 1.00 } },
  },

  .treeCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.2, 0.4, 0.6, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 1.0, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.4, 0.3, 0.3, 0.1, 0.1, 1.0, 1.0, 0.7, 0.5, 0.6, 0.5, 0.6, 0.6, 0.5, 0.6, 0.6, 0.6, 0.7, },
  },

  .ringCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.4, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.1, 0.3, 1.0, 1.0, 0.7, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.5, 0.4, 0.3, 0.3, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.8, 0.5, 0.1, 0.7, 0.2, 0.4, 0.4, 0.6, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },
};

static struct tuningModel rcclTuningModel[] = {
  tuning_model_0,
  tuning_model_1,
  tuning_model_2,
  tuning_model_3,
  tuning_model_4,
};

// LL128 max BW per channel
static const double ll128MaxBwPerCh = 20.0;
static const double llMaxBws[2][3] = { /* Volta-N1/Intel-N2/Intel-N4) */ {39.0, 39.0, 20.4}, /* Ampere-N1/AMD-N2/AMD-N4) */ {87.7, 22.5 /*avg of ring & tree*/, 19.0} };
static const double perChMaxTreeBws[2][3] = { /* Volta (N1/N2/N4) */ {26.5, 18.5, 10.0}, /* Ampere (N1/N2/N4) */ {24.0, 23.6, 17.8} };

ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph, struct ncclTopoGraph* collNetGraph) {
  int simpleDefaultThreads = (ringGraph->speedIntra*ringGraph->nChannels <= PCI_WIDTH) ? 256 : NCCL_SIMPLE_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] =
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 4*WARP_SIZE, NCCL_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 4*WARP_SIZE, NCCL_MAX_NTHREADS, NCCL_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_LL] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 4*WARP_SIZE, NCCL_MAX_NTHREADS, NCCL_MAX_NTHREADS);
#else
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_LL] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_LL_MAX_NTHREADS, NCCL_LL_MAX_NTHREADS);
#endif
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_LL128] =
    getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS/4, NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS);

  int nNodes = comm->nNodes;
  int nRanks = comm->nRanks;
  if (nRanks <= 1) return ncclSuccess;

  int compCap80 = minCompCap == 80 && maxCompCap == 80 ? 1 : 0;
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));
  int index2 = nNodes <= 2 ? nNodes-1 : 2;
  // LL: for single node, we look at GPU type; for multi-node, we look at CPU type
  int index1 = nNodes == 1 ? compCap80 : cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD ? 1 : 0;
  double llMaxBw = llMaxBws[index1][index2];
  double perChMaxTreeBw = perChMaxTreeBws[compCap80][index2];
  // De-penalize Tree/Simple latency on Power systems to favor Tree than Ring
  //if (cpuArch == NCCL_TOPO_CPU_ARCH_POWER) hwLat[NCCL_HW_PCI][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = hwLat[NCCL_HW_PCI][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
  float ppn = (float)nRanks / nNodes; // if ppn < 2, then we are sending/receiving at the same GPU through the NIC, apply some bw discount

  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = { treeGraph, ringGraph, collNetGraph };
  int intraHw[NCCL_NUM_ALGORITHMS], hw[NCCL_NUM_ALGORITHMS];
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) intraHw[a] = graphs[a]->typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) hw[a] = nNodes == 1 ? intraHw[a] : NCCL_HW_NET;

  for (int coll=0; coll<NCCL_NUM_FUNCTIONS; coll++) {
    int nsteps = coll == ncclFuncAllReduce ? 2*(nRanks-1) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nRanks-1 :
      nRanks;
    int nInterSteps = coll == ncclFuncAllReduce ? (nNodes > 1 ? 2*nNodes :0) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nNodes-1 :
      nNodes;

    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      if ((coll != ncclFuncAllReduce) && a != NCCL_ALGO_RING) continue;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        float speed = nNodes <= 2 || a == NCCL_ALGO_COLLNET ? graphs[a]->speedIntra : graphs[a]->speedInter;
        float busBw = comm->topo->baseBw != 0.0 ? comm->topo->baseBw : graphs[a]->nChannels * speed;

        // Various model refinements
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
        if (nNodes <= 2)
          busBw *= rcclTuningModel[comm->topo->tuning].bwRatio[0][a][p];
        else
          busBw *= rcclTuningModel[comm->topo->tuning].bwRatio[1][a][p];
#else
        if (compCap80) busBw = std::min(busBw, 235.0f);
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL) { busBw = std::min(llMaxBw, busBw * ((nNodes > 1 || coll == ncclFuncAllReduce || coll == ncclFuncReduce) ? 1.0/4.0 : 1.0/3.0)); }
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (ppn < 2 ? 0.7 : 0.92 /*120.0/128.0*/), ll128MaxBwPerCh*graphs[a]->nChannels);
        if (a == NCCL_ALGO_TREE) busBw = std::min(busBw*.92, graphs[a]->nChannels*perChMaxTreeBw);
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL) busBw = std::min(busBw*1.0/3.8, llMaxBw);
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (nNodes == 1 ? 7.0/9.0 : 120.0/128.0), ll128MaxBwPerCh*graphs[a]->nChannels);
#endif
        if (a == NCCL_ALGO_COLLNET && p != NCCL_PROTO_SIMPLE) busBw = 0;  // Oneshot CollNet only supports Simple

        // Convert bus BW to algorithm BW
        float ratio = (a != NCCL_ALGO_RING) ? .5 : (1.0 * nRanks) / nsteps;
        comm->bandwidths[coll][a][p] = busBw * ratio;

        comm->latencies[coll][a][p] = baseLat[a][p];
        float intraLat = rcclTuningModel[comm->topo->tuning].hwLat[intraHw[a]][a][p];
        float interLat = rcclTuningModel[comm->topo->tuning].hwLat[NCCL_HW_NET][a][p];
        //if (nNodes > 1 && p == NCCL_PROTO_LL) intraLat *= 1.8;
        if (a == NCCL_ALGO_RING) {
          float lat = rcclTuningModel[comm->topo->tuning].hwLat[hw[a]][a][p];
          if ((coll == ncclFuncReduce || coll == ncclFuncBroadcast)) {
            if (ringGraph->sameChannels) {
              comm->latencies[coll][a][p] += lat;
            } else {
              if (p == NCCL_PROTO_SIMPLE) lat = rcclTuningModel[comm->topo->tuning].hwLat[hw[a]][NCCL_ALGO_TREE][p]; // Add some chunk latency, waiting for proper chunk modeling
              comm->latencies[coll][a][p] += nsteps*lat;
            }
          } else {
            comm->latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;
          }
        } else if (a == NCCL_ALGO_TREE) {
          comm->latencies[coll][a][p] +=
            2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);
        } else {
          comm->latencies[coll][a][p] +=
            2 * (std::min(1, (nRanks/nNodes-1)) * intraLat + (nRanks/nNodes-1) * 0.5) + interLat;  // Add 0.5 arity serialization latency
        }
      }
    }
  }

  // Protocols/Algorithms enable/disable, and user overrides.
  // All are enabled except ll128 which is enabled by default only in certain cases.
  int protoEnable[NCCL_NUM_PROTOCOLS] = { 1, 2, 1 };
  int algoEnable[NCCL_NUM_ALGORITHMS] = { 1, 1, 1 };

  const char *protoStr = getenv("NCCL_PROTO");
  if (protoStr) {
    INFO(NCCL_ENV, "NCCL_PROTO set by environment to %s", protoStr);
    NCCLCHECK(parseList(protoStr, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoEnable));
  }
  const char *algoStr = getenv("NCCL_ALGO");
  if (algoStr) {
    INFO(NCCL_ENV, "NCCL_ALGO set by environment to %s", algoStr);
    NCCLCHECK(parseList(algoStr, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoEnable));
  }
  // Disable CollNet if it is not supported
  if (comm->collNetSupport == 0) {
    algoEnable[NCCL_ALGO_COLLNET] = 0;
    // If user has hard set NCCL_ALGO=COLLNET, ignore it
    if (algoEnable[NCCL_ALGO_RING] == 0 && algoEnable[NCCL_ALGO_TREE] == 0) {
      algoEnable[NCCL_ALGO_RING] = algoEnable[NCCL_ALGO_TREE] = 1;
      if (comm->rank == 0) WARN("CollNet is not supported or fails to initialize, ignoring NCCL_ALGO=COLLNET");
    }
  }

  for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    int pEnable = protoEnable[p];
    if (pEnable == 2 && p == NCCL_PROTO_LL128) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
      // Enable LL128 by default only on gfx90a with available tuning table
      pEnable = (graphs[a]->typeInter <= PATH_PXB) && graphs[a]->typeIntra <= PATH_NVL &&
        (comm->topo->nodes[GPU].nodes[0].gpu.gcn == 910 && comm->topo->ll128Enabled) ? 1 : 0;
#else
      // Enable LL128 by default only on Volta/Ampere+NVLink. Other cases are not tested and may cause silent data corruption.
      pEnable = (graphs[a]->typeInter <= PATH_PXB) && graphs[a]->typeIntra <= PATH_NVL &&
        ((minCompCap == 70 && maxCompCap == 70) || (minCompCap == 80 && maxCompCap == 80)) ? 1 : 0;
#endif
      if (comm->rank == 0 && c == 0 && a == 0) INFO(NCCL_INIT, "Using tuning table %d with LL128 %s", comm->topo->tuning, pEnable ? "enabled" : "disabled");
    }
    if (pEnable == 0) comm->bandwidths[c][a][p] = 0;
    // Only disable algo for Allreduce since others only have one
    if (c == ncclFuncAllReduce && algoEnable[a] == 0) comm->bandwidths[c][a][p] = 0;
  }

  if (comm->rank == 0) {
    char line[1024];
    sprintf(line, "Latency/AlgBw |");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        sprintf(line+strlen(line), " %7s/%6s |", ncclAlgoStr[a], ncclProtoStr[p]);
      }
    }
    INFO(NCCL_TUNING, "%s", line);
    sprintf(line, " Max NThreads |");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        sprintf(line+strlen(line), " %14d |", comm->maxThreads[a][p]);
      }
    }
    INFO(NCCL_TUNING, "%s", line);
    for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) {
      sprintf(line, "%13s |", ncclFuncStr[c]);
      for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
          sprintf(line+strlen(line), "%8.1f/%6.1f |", comm->latencies[c][a][p], comm->bandwidths[c][a][p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
    }
  }

  // Set per-thread amount of work before we increase nThreads and nChannels
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    comm->threadThresholds[a][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_LL128] = NCCL_LL128_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  }
  comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] *= nRanks;
  comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_SIMPLE] = 256;

  // Override defaults with user env
  char* str = getenv("NCCL_THREAD_THRESHOLDS");
  if (str) {
    INFO(NCCL_ENV, "NCCL_THREAD_THRESHOLDS set by environment to %s", str);
    ssize_t t[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {{ -2, -2, -2 }, { -2, -2, -2}};
    sscanf(str, "%ld %ld %ld %ld %ld %ld", t[0], t[0]+1, t[0]+2, t[1], t[1]+1, t[1]+2);
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (t[a][p] >= 0) comm->threadThresholds[a][p] = t[a][p];
      }
    }
  }

  INFO(NCCL_INIT, "threadThresholds %ld/%ld/%ld | %ld/%ld/%ld | %ld/%ld/%ld",
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_SIMPLE]);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetAlgoTime(struct ncclInfo* info, int algorithm, int protocol, int numPipeOps, float* time) {
  float bw = info->comm->bandwidths[info->coll][algorithm][protocol];
  float lat = info->comm->latencies[info->coll][algorithm][protocol];
  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  int logSize = log2i(info->nBytes>>6);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  if (algorithm == NCCL_ALGO_TREE) {
    if (logSize < 27) bw *= rcclTuningModel[info->comm->topo->tuning].treeCorrectionFactor[protocol][logSize];
    else bw *= rcclTuningModel[info->comm->topo->tuning].treeCorrectionFactor[protocol][26];
  }
  else if (algorithm == NCCL_ALGO_RING && info->comm->nNodes > 1) {
    if(logSize < 27) bw *= rcclTuningModel[info->comm->topo->tuning].ringCorrectionFactor[protocol][logSize];
    else bw *= rcclTuningModel[info->comm->topo->tuning].ringCorrectionFactor[protocol][26];
  }
#else
  if (algorithm == NCCL_ALGO_TREE && logSize < 23) bw *= treeCorrectionFactor[protocol][logSize];
  if (info->nChannels != 0) bw = bw / info->comm->nChannels * info->nChannels;
  if (algorithm == NCCL_ALGO_RING && protocol == NCCL_PROTO_SIMPLE && info->comm->nNodes > 1
      && info->coll == ncclFuncAllReduce && info->nBytes >= info->comm->nRanks/16.0*65536) lat *= 1.9; // Plateau effect of ring
#endif
  // Tree pipelining saves latency in aggregation cases
  int latCount = algorithm == NCCL_ALGO_RING ? numPipeOps : DIVUP(numPipeOps, NCCL_MAX_WORK_ELEMENTS);
  *time = lat * latCount + (info->nBytes) / (1000 * bw);
  return ncclSuccess;
}
