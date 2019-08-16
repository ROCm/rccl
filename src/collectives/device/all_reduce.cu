/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "all_reduce.h"
#include "collectives.h"

IMPL_COLL2(ncclAllReduce, sum,  FuncSum,  ncclCollAllReduce, ncclSum);
IMPL_COLL2(ncclAllReduce, prod, FuncProd, ncclCollAllReduce, ncclProd);
IMPL_COLL2(ncclAllReduce, min,  FuncMin,  ncclCollAllReduce, ncclMin);
IMPL_COLL2(ncclAllReduce, max,  FuncMax,  ncclCollAllReduce, ncclMax);