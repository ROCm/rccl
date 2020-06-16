/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gather.h"
#include "common.h"
#include "collectives.h"

IMPL_COLL_FUNC(ncclGather, copy, FuncSum, i8, int8_t);
IMPL_COLL_KERN(ncclGather, copy, FuncSum, i8, int8_t, 0);
