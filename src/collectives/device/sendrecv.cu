/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sendrecv.h"
#include "common.h"
#include "collectives.h"

IMPL_COLL_FUNC(ncclSendRecv, copy, FuncSum, i8, int8_t);
IMPL_COLL_KERN(ncclSendRecv, copy, FuncSum, i8, int8_t, 0);
