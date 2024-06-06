/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_STATUS_H_
#define MSCCL_STATUS_H_

#include "msccl/msccl_struct.h"

void mscclSetThreadRank(int rank);

bool& mscclInitialized();

mscclStatus& mscclGetStatus();

mscclSavedProxyArgs& mscclGetSavedProxyArgs();

mscclThreadLocalStatus& mscclGetThreadLocalStatus();

#endif
