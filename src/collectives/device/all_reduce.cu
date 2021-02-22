/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
*  Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 * See LICENSE.txt for license information
 ************************************************************************/

#include "all_reduce.h"
#include "common.h"
#include "collectives.h"

// [RCCL]
// IMPL_COLL_R(AllReduce);
IMPL_COLL_CLIQUE(AllReduce);
// [/RCCL]
