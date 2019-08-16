/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "reduce.h"
#include "collectives.h"

IMPL_COLL2(ncclReduce, sum,  FuncSum,  ncclCollReduce, ncclSum);
IMPL_COLL2(ncclReduce, prod, FuncProd, ncclCollReduce, ncclProd);
IMPL_COLL2(ncclReduce, min,  FuncMin,  ncclCollReduce, ncclMin);
IMPL_COLL2(ncclReduce, max,  FuncMax,  ncclCollReduce, ncclMax);
