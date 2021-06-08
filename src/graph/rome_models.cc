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
#include "core.h"
#include "graph.h"
#include "topo.h"
#include "xml.h"
#include <math.h>
#include <sys/time.h>
#include "rome_models.h"

struct rcclRomeModel {
  int nGpus;
  int nCpus;
  int nNics;
  int nLinks;
  int64_t gpuIds[NCCL_TOPO_MAX_NODES];
  int64_t nicIds[NCCL_TOPO_MAX_NODES];
  int64_t gpuNuma[NCCL_TOPO_MAX_NODES];
  int64_t nicNuma[NCCL_TOPO_MAX_NODES];
  uint8_t connMatrix[NCCL_TOPO_MAX_NODES*NCCL_TOPO_MAX_NODES];
  const char *pattern;
  const char *ringBase;
};

static struct rcclRomeModel rome_model_22 = {
  .nGpus = 8, .nCpus = 4, .nNics = 1, .nLinks = 2,
  .gpuIds = { 0x3000, 0x43000, 0x26000, 0xc3000, 0x83000, 0x23000, 0xc6000, 0xa3000, },
  .nicIds = { 0xe1000, },
  .gpuNuma = { 1, 0, 1, 2, 3, 1, 2, 3, },
  .nicNuma = { 2, },
  .connMatrix = { 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, },
  .pattern = "10302120",
  .ringBase = "7 4 5 3 1 0 6 2|4 7 3 5 0 1 2 6",
};

static struct rcclRomeModel rome_model_25 = {
  .nGpus = 8, .nCpus = 4, .nNics = 2, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { 0x61000, 0xa1000, },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 2, 3, },
  .nicNuma = { 0, 3, },
  .connMatrix = { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, },
  .pattern = "11303011",
  .ringBase = "2 1 0 3 6 7 5 4|7 6 4 5 1 2 3 0",
};

static struct rcclRomeModel rome_model_27 = {
  .nGpus = 8, .nCpus = 4, .nNics = 2, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { 0x61000, 0xa1000, },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 2, 3, },
  .nicNuma = { 0, 3, },
  .connMatrix = { 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, },
  .pattern = "11303011",
  .ringBase = "0 6 2 3 1 7 5 4|7 1 4 5 6 0 3 2",
};

static struct rcclRomeModel rome_model_29 = {
  .nGpus = 8, .nCpus = 4, .nNics = 1, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { 0xe1000, },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { 2, },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .pattern = "10302120",
  .ringBase = "6 5 7 4 0 1 3 2|6 4 7 5 2 3 1 0",
};

static struct rcclRomeModel rome_model_31 = {
  .nGpus = 8, .nCpus = 8, .nNics = 2, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { 0x61000, 0xa1000, },
  .gpuNuma = { 1, 2, 2, 3, 4, 5, 5, 7, },
  .nicNuma = { 0, 6, },
  .connMatrix = { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, },
  .pattern = "0110201010200110",
  .ringBase = "1 2 3 0 6 4 5 7|4 6 7 5 2 1 0 3",
};

static struct rcclRomeModel rome_model_33 = {
  .nGpus = 8, .nCpus = 8, .nNics = 2, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { 0x61000, 0xa1000, },
  .gpuNuma = { 1, 2, 2, 3, 4, 5, 5, 7, },
  .nicNuma = { 0, 6, },
  .connMatrix = { 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, },
  .pattern = "0110201010200110",
  .ringBase = "1 4 5 7 0 3 2 6|4 1 7 5 6 2 3 0",
};

static struct rcclRomeModel rome_model_30 = {
  .nGpus = 8, .nCpus = 8, .nNics = 0, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 1, 2, 2, 3, 4, 5, 5, 7, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, },
  .pattern = "0010201010200010",
  .ringBase = "3 0 1 2 6 7 5 4|2 1 0 3 7 6 4 5",
};

static struct rcclRomeModel rome_model_32 = {
  .nGpus = 8, .nCpus = 8, .nNics = 0, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 1, 2, 2, 3, 4, 5, 5, 7, },
  .nicNuma = { },
  .connMatrix = { 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, },
  .pattern = "0010201010200010",
  .ringBase = "0 6 2 3 4 5 7 1|3 2 6 0 1 7 5 4",
};

static struct rcclRomeModel rome_model_24 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 2, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, },
  .pattern = "10303010",
  .ringBase = "0 1 2 3 5 7 6 4|1 0 3 2 7 5 4 6",
};

static struct rcclRomeModel rome_model_26 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 2, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, },
  .pattern = "10303010",
  .ringBase = "4 5 7 1 0 3 2 6|3 0 6 2 1 7 5 4",
};

static struct rcclRomeModel rome_model_23 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, },
  .pattern = "10302020",
  .ringBase = "1 7 6 4 5 2 0 3|2 5 3 0 4 6 7 1",
};

static struct rcclRomeModel rome_model_38 = {
  .nGpus = 8, .nCpus = 7, .nNics = 0, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 1, 2, 2, 3, 5, 5, 6, 7, },
  .nicNuma = { },
  .connMatrix = { 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, },
  .pattern = "10201000201010",
  .ringBase = "6 7 1 4 3 5 2 0|0 2 5 3 4 1 7 6",
};

static struct rcclRomeModel rome_model_28 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .pattern = "10302020",
  .ringBase = "0 3 2 1 4 5 6 7|7 6 5 4 1 2 3 0|0 2 5 7 4 6 3 1|1 3 6 4 7 5 2 0",
};

static struct rcclRomeModel rome_model_40 = {
  .nGpus = 8, .nCpus = 4, .nNics = 1, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { 0xe1000, },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { 2, },
  .connMatrix = { 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, },
  .pattern = "10302120",
  .ringBase = "6 7 1 4 0 5 3 2|7 6 4 1 0 2 3 5",
};

static struct rcclRomeModel rome_model_42 = {
  .nGpus = 8, .nCpus = 7, .nNics = 1, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { 0xe1000, },
  .gpuNuma = { 1, 2, 2, 3, 5, 5, 6, 7, },
  .nicNuma = { 4, },
  .connMatrix = { 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, },
  .pattern = "10201001201010",
  .ringBase = "7 4 6 1 3 0 2 5|6 4 7 1 3 2 5 0",
};

static struct rcclRomeModel rome_model_44 = {
  .nGpus = 8, .nCpus = 4, .nNics = 1, .nLinks = 3,
  .gpuIds = { 0x63000, 0x43000, 0x27000, 0x3000, 0xe3000, 0xc3000, 0xa3000, 0x83000, },
  .nicIds = { 0xc4000, },
  .gpuNuma = { 0, 0, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { 2, },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .pattern = "20202120",
  .ringBase = "5 4 7 6 2 1 3 0|5 6 7 4 1 0 2 3",
};

static struct rcclRomeModel rome_model_45 = {
  .nGpus = 8, .nCpus = 7, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 1, 2, 2, 3, 5, 5, 6, 7, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .pattern = "10201000201010",
  .ringBase = "0 1 2 3 4 5 6 7|0 2 5 7 4 6 1 3|0 3 1 6 4 7 5 2|0 7 6 5 4 3 2 1",
};

static struct rcclRomeModel rome_model_46 = {
  .nGpus = 8, .nCpus = 7, .nNics = 1, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { 0xe1000, },
  .gpuNuma = { 1, 2, 2, 3, 5, 5, 6, 7, },
  .nicNuma = { 4, },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .pattern = "10201001201010",
  .ringBase = "6 5 7 4 1 2 3 0|7 4 6 5 1 0 3 2",
};

static struct rcclRomeModel rome_model_48 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0x4a000, 0x50000, 0xa000, 0xf000, 0xcb000, 0xd1000, 0x8a000, 0x90000, },
  .nicIds = { },
  .gpuNuma = { 0, 0, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .pattern = "20202020",
  .ringBase = "0 1 2 3 4 5 6 7|7 6 5 4 3 2 1 0|0 1 2 3 4 5 6 7|7 6 5 4 3 2 1 0",
};

static struct rcclRomeModel rome_model_49 = {
  .nGpus = 8, .nCpus = 4, .nNics = 4, .nLinks = 3,
  .gpuIds = { 0x4a000, 0x50000, 0xa000, 0xf000, 0xcb000, 0xd1000, 0x8a000, 0x90000, },
  .nicIds = { 0x45000, 0x13000, 0xc6000, 0x85000, },
  .gpuNuma = { 0, 0, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { 0, 1, 2, 3, },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .pattern = "21212121",
  .ringBase = "N0 0 1 2 3 4 5 6 7 N3|N3 7 6 5 4 3 2 1 0 N0|N1 2 3 0 1 6 7 4 5 N2|N2 5 4 7 6 1 0 3 2 N1",
};

static struct rcclRomeModel rome_model_52 = {
  .nGpus = 8, .nCpus = 1, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0xc1000, 0xc5000, 0xc9000, 0xcd000, 0xd1000, 0xd5000, 0xd9000, 0xdd000, },
  .nicIds = { },
  .gpuNuma = { 0, 0, 0, 0, 0, 0, 0, 0, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, },
  .pattern = "80",
  .ringBase = "0 1 3 2 4 5 7 6|6 7 5 4 2 3 1 0|0 1 5 4 6 7 3 2|2 3 7 6 4 5 1 0",
};

static struct rcclRomeModel rome_model_53 = {
  .nGpus = 8, .nCpus = 4, .nNics = 4, .nLinks = 3,
  .gpuIds = { 0x4a000, 0x50000, 0xa000, 0xf000, 0xcb000, 0xd1000, 0x8a000, 0x90000, },
  .nicIds = { 0x45000, 0x13000, 0xc6000, 0x85000, },
  .gpuNuma = { 1, 1, 3, 3, 5, 5, 7, 7, },
  .nicNuma = { 1, 3, 5, 7, },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .pattern = "21212121",
  .ringBase = "N0 0 1 2 3 4 5 6 7 N3|N3 7 6 5 4 3 2 1 0 N0|N1 2 3 0 1 6 7 4 5 N2|N2 5 4 7 6 1 0 3 2 N1",
};

static struct rcclRomeModel rome_model_43 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0x63000, 0x43000, 0x27000, 0x3000, 0xe3000, 0xc3000, 0xa3000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 0, 0, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .pattern = "20202020",
  .ringBase = "0 1 2 3 4 5 6 7|0 2 5 7 4 6 1 3|0 3 1 6 4 7 5 2|0 7 6 5 4 3 2 1",
};

static struct rcclRomeModel romeTopoModels[] = {
  rome_model_22,
  rome_model_25,
  rome_model_27,
  rome_model_29,
  rome_model_31,
  rome_model_33,
  rome_model_30,
  rome_model_32,
  rome_model_24,
  rome_model_26,
  rome_model_23,
  rome_model_38,
  rome_model_28,
  rome_model_40,
  rome_model_42,
  rome_model_44,
  rome_model_45,
  rome_model_46,
  rome_model_48,
  rome_model_49,
  rome_model_52,
  rome_model_53,
  rome_model_43,
};

/* Parse user defined rings. Format is like :
 * "0 1|1 0|0 1 2 3|3 2 1 0|N0 0 2 3 1 N1|1 3 2 0|0 1 2 3 4 5 6 7|N2 7 6 5 4 3 2 1 0 N1"
 * Network interfaces can be optionally specified by N prefix.
 * Rings with a non-matching number of gpus are ignored so we can provide
 * rings for multiple cases.
 */
ncclResult_t parseGraph(const char* str, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* gpu_map) {
  int gpus[NCCL_TOPO_MAX_NODES];
  int nChannels = 0;
  int gpu = 0;
  int offset = 0;
  int status = 0; // 0 : between numbers, 1 : inside number, 2: start NET, 3: inside NET
  int nets[NCCL_TOPO_MAX_NODES*2];
  int net_offset = 0, net_count = 0;
  int ngpus = system->nodes[GPU].count;
  int nnets = system->nodes[NET].count;
  do {
    if (str[offset] == 'N') {
      if (status == 0) {
        status = 2;
      }
    } else {
      int digit = str[offset] - '0';
      if (digit >= 0 && digit <= 9) {
        switch (status) {
          case 0:
            gpus[gpu] = digit;
            status = 1;
            break;
          case 1:
            gpus[gpu] = gpus[gpu]*10+digit;
            break;
          case 2:
            nets[net_offset] = digit+'N';
            status = 3;
            break;
          case 3:
            nets[net_offset] = (nets[net_offset]-'N')*10+digit+'N';
            break;
        }
      } else {
        if (status == 1) {
          gpu++;
          net_offset = 2*gpu-1;
          if (gpu > NCCL_TOPO_MAX_NODES) goto end;
        } else if (status == 2 || status == 3) {
          net_offset++;
          net_count++;
          if (net_offset > ngpus*2) goto end;
        }
        status = 0;
        if (str[offset] == '|' || str[offset] == '\0') {
          // Ignore if ngpus doesn't match
          if (gpu != ngpus) goto newchannel;
          // Ignore if net_count is not 0 or odd number
          if (net_count && net_count%2) goto newchannel;

          for (int r=0; r<ngpus; r++) {
            int g = gpus[r];
            // Ignore if gpus are out of bounds
            if (g < 0 || g >= ngpus) goto newchannel;
            // Ignore if gpus are duplicate
            for (int i=0; i<r; i++)
              if (gpus[i] == g) goto newchannel;
            // remap if needed
            if (gpu_map) g = gpu_map[g];
            // Translate gpu numbers into ranks
            int j = 0;
            for (j = 0; j < ngpus; j++)
              if (g == system->nodes[GPU].nodes[j].gpu.dev)
                break;
            if (j < ngpus)
              graph->intra[nChannels*ngpus+r] = system->nodes[GPU].nodes[j].gpu.rank;
            else
              return ncclInternalError;
          }

          if (net_count) {
            memcpy(&graph->intraNets[ngpus*nChannels*2], nets, ngpus*2*sizeof(int));
            graph->nIntraChannels++;
            if (nets[0]-'N' >= nnets || nets[ngpus*2-1]-'N' >= nnets) goto newchannel;
            graph->inter[nChannels*2] = nets[0]-'N';
            graph->inter[nChannels*2+1] = nets[ngpus*2-1]-'N';
          } else if (nnets) {
            graph->inter[nChannels*2] = system->nodes[NET].nodes[nChannels%nnets].id;
            graph->inter[nChannels*2+1] = system->nodes[NET].nodes[(nChannels+1)%nnets].id;
          }
          nChannels++;
newchannel:
          gpu = 0;
          net_offset = 0;
          net_count = 0;
        }
      }
    }
  } while (str[offset++] != 0);
end:
  graph->nChannels = nChannels;
  graph->speedIntra = graph->speedInter = system->maxWidth;
#if 0
  for (int i=0; i<graph->nChannels; i++) {
    printf("%d: ", i);
    printf ("NET/%d ", graph->inter[i*2]);
    for (int j=0; j<ngpus; j++) printf("GPU/%d ", graph->intra[i*ngpus+j]);
    printf ("NET/%d ", graph->inter[i*2+1]);
    printf("\n");
  }
#endif
  return ncclSuccess;
}

ncclResult_t parseChordalRing(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  static const char *ringBase = "0 1 2 3 5 4 7 6|0 2 4 1 7 3 6 5|0 3 1 5 7 2 6 4|0 6 7 4 5 3 2 1|0 5 6 3 7 1 4 2|0 4 6 2 7 5 1 3";
  int id[8], dist[8];
  int i;

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
  system->type |= RCCL_TOPO_CR8G;
  NCCLCHECK(parseGraph(ringBase, system, graph, id));
  if (system->nodes[NET].count && system->nodes[GPU].count != system->nRanks) {
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
  return ncclSuccess;
}

struct ncclGpuIdHIP {
  int g;
  int dev;
};

static int cmpIds(const void * g1, const void * g2) {
  struct ncclGpuIdHIP *s1 = (struct ncclGpuIdHIP*)g1;
  struct ncclGpuIdHIP *s2 = (struct ncclGpuIdHIP*)g2;
  return s1->dev - s2->dev;
}

struct ncclCpuNuma {
  int c;
  uint64_t numa;
};

static int cmpNuma(const void * g1, const void * g2) {
  struct ncclCpuNuma *s1 = (struct ncclCpuNuma*)g1;
  struct ncclCpuNuma *s2 = (struct ncclCpuNuma*)g2;
  return s1->numa - s2->numa;
}

static ncclResult_t parseRomeSystem(struct ncclTopoSystem* system, struct rcclRomeModel* romeTopo, char *pattern) {
  pattern[0] = 0; // pattern will be NULL for invalid topology
  romeTopo->nGpus = system->nodes[GPU].count;
  romeTopo->nCpus = system->nodes[CPU].count;
  romeTopo->nNics = system->nodes[NET].count;
  romeTopo->nLinks = 0;
  // sort GPU devices by HIP device ID
  struct ncclGpuIdHIP scores[NCCL_TOPO_MAX_NODES];
  for (int i = 0; i < romeTopo->nGpus; i ++) {
    scores[i].g = i;
    scores[i].dev = system->nodes[GPU].nodes[i].gpu.dev;
  }
  qsort(scores, romeTopo->nGpus, sizeof(struct ncclGpuIdHIP), cmpIds);
  // sort CPU devices by NUMA id
  struct ncclCpuNuma cpu_scores[NCCL_TOPO_MAX_NODES];
  for (int i = 0; i < romeTopo->nCpus; i ++) {
    cpu_scores[i].c = i;
    cpu_scores[i].numa = system->nodes[CPU].nodes[i].id;
  }
  qsort(cpu_scores, romeTopo->nCpus, sizeof(struct ncclCpuNuma), cmpNuma);

  for (int i = 0; i < romeTopo->nGpus; i ++) {
    int gpu, n, m, distance;
    gpu = scores[i].g;
    romeTopo->gpuIds[i] = system->nodes[GPU].nodes[gpu].id;
    m = 0;
    distance = system->nodes[GPU].nodes[gpu].paths[CPU][m].count;
    for (n = 1; n < romeTopo->nCpus; n++) {
      if (system->nodes[GPU].nodes[gpu].paths[CPU][n].count < distance) {
        distance = system->nodes[GPU].nodes[gpu].paths[CPU][n].count;
        m = n;
      }
    }
    if (m < romeTopo->nCpus) romeTopo->gpuNuma[i] = system->nodes[CPU].nodes[m].id;

    struct ncclTopoNode* node = system->nodes[GPU].nodes+gpu;
    if (node->paths[GPU] == NULL) continue;
    int count = 0;
    for (n = 0; n < romeTopo->nGpus; n++) {
      romeTopo->connMatrix[i*romeTopo->nGpus+n] = 0;
      struct ncclTopoLink* link;
      for (link = node->links; link->remNode; link++) {
        if (link->remNode->gpu.dev == n) break;
      }
      if (!link->remNode) continue;
      if (link->type != LINK_NVL) continue;
      romeTopo->connMatrix[i*romeTopo->nGpus+n] = 1;
      count ++;
    }
    if (romeTopo->nLinks < count) romeTopo->nLinks = count;
  }

  for (int net = 0; net < romeTopo->nNics; net++) {
    int n, m, distance;
    m = 0;
    romeTopo->nicIds[net] = system->nodes[NET].nodes[net].net.busId;
    distance = system->nodes[NET].nodes[net].paths[CPU][m].count;
    for (n = 0; n < romeTopo->nCpus; n++)
      if (system->nodes[NET].nodes[net].paths[CPU][n].count < distance) {
        distance = system->nodes[NET].nodes[net].paths[CPU][n].count;
        m = n;
      }
    if (m < romeTopo->nCpus) romeTopo->nicNuma[net] = system->nodes[CPU].nodes[m].id;
    else return ncclSuccess;
  }

  // number of GPUs and NICs on each numa node is used as first screening pattern
  for (int i = 0; i < romeTopo->nCpus; i++) {
    uint64_t id = system->nodes[CPU].nodes[cpu_scores[i].c].id;
    int g = 0, n = 0;
    for (int j = 0; j < romeTopo->nGpus; j++)
      if (romeTopo->gpuNuma[j] == id) g++;
    for (int j = 0; j < romeTopo->nNics; j++)
      if (romeTopo->nicNuma[j] == id) n++;
    pattern[i*2] = '0' + g;
    pattern[i*2+1] = '0' + n;
  }
  pattern[romeTopo->nCpus*2] = 0;

  const char* romeModelFile = getenv("RCCL_DUMP_ROME_MODEL_FILE");
  if (romeModelFile) {
    INFO(NCCL_ENV, "RCCL_DUMP_ROME_MODEL_FILE set by environment to %s", romeModelFile);
    FILE* file = fopen(romeModelFile, "w");
    if (file == NULL) {
      WARN("Unable to open %s, not dumping Rome model.", romeModelFile);
      return ncclSuccess;
    }
    fprintf(file, "static struct rcclRomeModel rome_model_ = {\n");
    fprintf(file, "  .nGpus = %d, .nCpus = %d, .nNics = %d, .nLinks = %d,\n", romeTopo->nGpus, romeTopo->nCpus, romeTopo->nNics, romeTopo->nLinks);
    fprintf(file, "  .gpuIds = { ");
    for (int i = 0; i < romeTopo->nGpus; i ++) fprintf(file, "0x%lx, ", romeTopo->gpuIds[i]);
    fprintf(file, "},\n");
    fprintf(file, "  .nicIds = { ");
    for (int i = 0; i < romeTopo->nNics; i ++) fprintf(file, "0x%lx, ", romeTopo->nicIds[i]);
    fprintf(file, "},\n");
    fprintf(file, "  .gpuNuma = { ");
    for (int i = 0; i < romeTopo->nGpus; i ++) fprintf(file, "%ld, ", romeTopo->gpuNuma[i]);
    fprintf(file, "},\n");
    fprintf(file, "  .nicNuma = { ");
    for (int i = 0; i < romeTopo->nNics; i ++) fprintf(file, "%ld, ", romeTopo->nicNuma[i]);
    fprintf(file, "},\n");
    fprintf(file, "  .connMatrix = { ");
    for (int i = 0; i < romeTopo->nGpus; i ++)
      for (int n = 0; n < romeTopo->nGpus; n++) fprintf(file, "%d, ", romeTopo->connMatrix[i*romeTopo->nGpus+n]);
    fprintf(file, "},\n");
    fprintf(file, "  .pattern = \"%s\",\n", pattern);
    fprintf(file, "  .ringBase = \"\",\n");
    fprintf(file, "};\n");
    fclose(file);
  }
  return ncclSuccess;
}

static bool permuteGpuIds(int *g, int n, int last, struct rcclRomeModel* ref, struct rcclRomeModel* topo, int* time, bool nbio) {
  (*time) ++;
  if (n == last) {
    int i, j;
    // match GPU numa
    for (i = 0; i < ref->nGpus; i++)
      if (ref->gpuNuma[i] != topo->gpuNuma[g[i]]) break;
    if (i < ref->nGpus) return false;
    // match XGMI connection
    for (i = 0; i < ref->nGpus; i++) {
      for (j = 0; j < ref->nGpus; j++) {
        if (ref->connMatrix[i*ref->nGpus+j] != topo->connMatrix[g[i]*ref->nGpus+g[j]]) break;
        if ((ref->gpuIds[i]-ref->gpuIds[j])*(topo->gpuIds[g[i]]-topo->gpuIds[g[j]]) < 0) break;
      }
      if (j < ref->nGpus) break;
    }
    if (i < ref->nGpus) return false;
    // match NBIO
    if (nbio) {
      for (i = 0; i < ref->nGpus; i++) {
        for (j = 0; j < ref->nGpus; j++) {
          if (i == j) continue;
          bool nbio_ref = (ref->gpuIds[i]&0xf0000) == (ref->gpuIds[j]&0xf0000);
          bool nbio_topo = (topo->gpuIds[g[i]]&0xf0000) == (topo->gpuIds[g[j]]&0xf0000);
          if (nbio_ref != nbio_topo) break;
          if (nbio_ref && ((ref->gpuIds[i]-ref->gpuIds[j])*(topo->gpuIds[g[i]]-topo->gpuIds[g[j]]) < 0)) break;
        }
        if (j < ref->nGpus) break;
      }
      if (i < ref->nGpus) return false;
    }
    return true;
  } else {
    for (int i = n; i <= last; i++) {
      std::swap(g[n], g[i]);
      if (permuteGpuIds(g, n+1, last, ref, topo, time, nbio)) return true;
      std::swap(g[n], g[i]);
    }
  }
  return false;
}

ncclResult_t parseRome4P2H(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  static char ringRemap[64];
  int i;

  int ngpus = system->nodes[GPU].count;
  int ncpus = system->nodes[CPU].count;

  // only valid on Rome
  int arch, vendor, model;
  NCCLCHECK(ncclTopoCpuType(system, &arch, &vendor, &model));
  if (arch != NCCL_TOPO_CPU_ARCH_X86 || vendor != NCCL_TOPO_CPU_VENDOR_AMD || model != NCCL_TOPO_CPU_TYPE_ROME)
    return ncclSuccess;

  // number of GPUs and NICs on each numa node is used as first screening pattern
  struct rcclRomeModel romeTopo;
  char pattern[256];
  NCCLCHECK(parseRomeSystem(system, &romeTopo, pattern));

  // recognize system as Rome 4P2H even if no matching model
  if (ngpus > 4 && romeTopo.nLinks) system->type |= RCCL_TOPO_4P2H_ROME;

  int g[NCCL_TOPO_MAX_NODES];
  int time = 0;
  struct timeval tvs, tve;
  gettimeofday(&tvs, NULL);

  // check if GPUs are directly connected to CPU
  bool match_nbio = true;
  for (i = 0; i < romeTopo.nGpus; i++) {
    int cpu, gpu;
    NCCLCHECK(ncclTopoIdToIndex(system, CPU,  romeTopo.gpuNuma[i], &cpu));
    NCCLCHECK(ncclTopoIdToIndex(system, GPU,  romeTopo.gpuIds[i], &gpu));
    if (system->nodes[GPU].nodes[gpu].paths[CPU][cpu].count > 2) break;
  }
  if (i < romeTopo.nGpus) match_nbio = false;

  for (i = 0; i < sizeof(romeTopoModels)/sizeof(romeTopoModels[0]); i++) {
    if (romeTopo.nCpus != romeTopoModels[i].nCpus || romeTopo.nGpus != romeTopoModels[i].nGpus ||
      romeTopo.nNics != romeTopoModels[i].nNics || romeTopo.nLinks != romeTopoModels[i].nLinks) continue;
    if (strcmp(romeTopoModels[i].pattern, pattern)) continue;
    for (int j = 0; j < ngpus; j++) g[j] = (j+2)%ngpus;
    if (permuteGpuIds(g, 0, ngpus-1, romeTopoModels+i, &romeTopo, &time, match_nbio)) break;
  }
  gettimeofday(&tve, NULL);
  float t = (tve.tv_sec - tvs.tv_sec)*1E3 + (tve.tv_usec - tvs.tv_usec)/1E3;
  if (i >= sizeof(romeTopoModels)/sizeof(romeTopoModels[0])) {
    //printf("No solution in %.2fms (%d iter)\n", t, time);
    return ncclSuccess;
  }

  char line[1024];
  //sprintf(line, "Found matching Rome model index %d in %.2fms (%d iter) with GPU mapping: ", i, t, time);
  sprintf(line, "Found matching Rome model index %d with GPU mapping: ", i);
  int offset = strlen(line);
  for (int k = 0; k < ngpus; k++) {
    sprintf(line+offset, "%d ", g[k]);
    offset = strlen(line);
  }
  INFO(NCCL_GRAPH, "%s", line);

  // create 4P2H based on reference and remapped ids
  NCCLCHECK(parseGraph(romeTopoModels[i].ringBase, system, graph, g));
  return ncclSuccess;
}
