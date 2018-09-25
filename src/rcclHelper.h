/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclHelper.h
 * @brief Contains signatures for cross-stream synchronization
 *
 * This file contains signatures for cross-stream synchronization when same
 * communicator is used on different streams.
 */

#pragma once

#include <hip/hip_runtime_api.h>
#include "rcclTracker.h"

//! Record event on the stream before launching kernels related to op
void PreEnqueueEventRecord(RcclComm_t*, hipStream_t);

//! Record event on the stream after launching kernels related to op
void PostEnqueueEventRecord(RcclComm_t*, hipStream_t);
