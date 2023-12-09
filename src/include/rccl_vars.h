/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef RCCL_VARS_H_
#define RCCL_VARS_H_

#include "param.h"

RCCL_PARAM_DECLARE(EnableHipGraph);      // Opt-in environment variable for enabling hipGraph
RCCL_PARAM_DECLARE(ExperimentalXccMode); // Opt-in environment variable for enabling experimental XCC filtering

int32_t __inline__ rcclGetPreferredXcc(int src, int dst)
{
  constexpr int32_t table1[8][8] = {{0,1,5,6,4,3,7,2},
                                    {1,0,7,5,6,2,4,3},
                                    {6,7,0,1,3,4,2,5},
                                    {5,6,1,0,2,7,3,4},
                                    {4,6,3,2,0,5,1,7},
                                    {3,2,4,7,5,0,6,1},
                                    {7,4,2,3,1,5,0,6},
                                    {2,3,5,4,7,1,6,0}};

  constexpr int32_t table2[8][8] = {{0,1,4,6,2,7,3,5},
                                    {1,0,4,6,2,5,3,7},
                                    {4,6,0,1,5,2,7,3},
                                    {6,7,1,0,4,2,5,3},
                                    {2,3,4,5,0,6,1,7},
                                    {6,4,2,3,7,0,5,1},
                                    {2,3,6,4,1,7,0,5},
                                    {4,6,2,3,5,1,7,0}};

  if (src < 0 || src >= 8 || dst < 0 || dst >= 8) return -1;

  switch(rcclParamExperimentalXccMode())
  {
  case 1: return table1[src][dst];
  case 2: return table2[src][dst];
  default: return -1;
  }
}

#endif
