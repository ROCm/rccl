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

#define MAX_ROME_GPUS 16
#define MAX_ROME_NICS 8

struct rcclRomeModel {
  int nGpus;
  int nCpus;
  int nNics;
  int nLinks;
  int64_t gpuIds[MAX_ROME_GPUS];
  int64_t nicIds[MAX_ROME_NICS];
  int64_t gpuNuma[MAX_ROME_GPUS];
  int64_t nicNuma[MAX_ROME_NICS];
  int connMatrix[MAX_ROME_GPUS*MAX_ROME_GPUS];
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
  .pattern = "00102010002010",
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
  .pattern = "00102010012010",
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
  .pattern = "00102010002010",
  .ringBase = "0 1 2 3 4 5 6 7|0 2 5 7 4 6 1 3|0 3 1 6 4 7 5 2|0 7 6 5 4 3 2 1",
};

static struct rcclRomeModel rome_model_46 = {
  .nGpus = 8, .nCpus = 7, .nNics = 1, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { 0xe1000, },
  .gpuNuma = { 1, 2, 2, 3, 5, 5, 6, 7, },
  .nicNuma = { 4, },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .pattern = "00102010012010",
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

static struct rcclRomeModel rome_model_50 = {
  .nGpus = 8, .nCpus = 2, .nNics = 1, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { 0xe1000, },
  .gpuNuma = { 0, 0, 0, 0, 1, 1, 1, 1, },
  .nicNuma = { 1, },
  .connMatrix = { 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, },
  .pattern = "4041",
  .ringBase = "1 7 6 4 5 2 0 3|2 5 3 0 4 6 7 1",
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
  rome_model_50,
};
