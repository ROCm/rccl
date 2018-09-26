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

//! Synchronize current stream with stream used before with the same
//! communicator. If previous stream is same as current stream, don't do
//! anything

//! \param [in] comm Memory location to internal Rccl communicator
//! \param [in] stream Stream with which the op will be synchronized with
void PreEnqueueEventRecord(RcclComm_t* comm, hipStream_t stream);

//! Record event on the stream after launching kernels related to op

//! \param [in] comm Memory location to internal Rccl communicator
//! \param [in] stream Stream with which the op will be synchronized with
void PostEnqueueEventRecord(RcclComm_t* comm, hipStream_t stream);
