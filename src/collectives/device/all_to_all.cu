/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "all_to_all.h"
#include "common.h"
#include "collectives.h"

IMPL_COLL_FUNC(ncclAllToAll, copy, FuncSum, i8, int8_t);
IMPL_COLL_KERN(ncclAllToAll, copy, FuncSum, i8, int8_t, 0);
