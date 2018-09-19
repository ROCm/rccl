/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include <hip/hip_runtime_api.h>
#include "rcclTracker.h"

void PreEnqueueEventRecord(RcclComm_t*, hipStream_t);
void PostEnqueueEventRecord(RcclComm_t*, hipStream_t);
