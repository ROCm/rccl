/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "all_gather.h"
#include "collectives.h"

IMPL_COLL3(ncclAllGather, copy, FuncSum, i8, int8_t, ncclCollAllGather, ncclSum, ncclInt8);
