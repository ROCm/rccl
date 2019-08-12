/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "broadcast.h"
#include "collectives.h"

#define UNROLL 4

IMPL_COLL3(ncclBroadcast, copy, FuncSum, i8, int8_t, ncclCollBroadcast, ncclSum, ncclInt8);
