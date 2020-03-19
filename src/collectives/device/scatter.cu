/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "scatter.h"
#include "common.h"
#include "collectives.h"

IMPL_COLL_FUNC(ncclScatter, copy, FuncSum, i8, int8_t);
IMPL_COLL_KERN(ncclScatter, copy, FuncSum, i8, int8_t, 0);
