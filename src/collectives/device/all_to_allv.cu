/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "all_to_allv.h"
#include "common.h"
#include "collectives.h"

IMPL_COLL_FUNC(ncclAllToAllv, copy, FuncSum, i8, int8_t);
IMPL_COLL_KERN(ncclAllToAllv, copy, FuncSum, i8, int8_t, 0);
