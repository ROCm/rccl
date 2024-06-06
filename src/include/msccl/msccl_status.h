/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_STATUS_H_
#define MSCCL_STATUS_H_

#include "msccl/msccl_struct.h"

void mscclSetThreadLocalComm(struct ncclComm*);

mscclStatus& mscclGetStatus();

mscclSavedProxyArgs& mscclGetSavedProxyArgs();

mscclThreadLocalStatus& mscclGetThreadLocalStatus();

#endif
