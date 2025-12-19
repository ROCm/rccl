/* Copyright (c) Advanced Micro Devices, Inc. */
/* RCCL HIP Tracer - HIP call type IDs */

#ifndef RCCL_HIP_API_H
#define RCCL_HIP_API_H

// HIP call type IDs (matches rcclCall_t enum values in RCCL)
#define RCCL_HIP_STREAM_SYNCHRONIZE  0
#define RCCL_HIP_DEVICE_SYNCHRONIZE  1
#define RCCL_HIP_EVENT_SYNCHRONIZE   2
#define RCCL_HIP_EVENT_RECORD        3
#define RCCL_HIP_STREAM_WAIT_EVENT   4
#define RCCL_HIP_EVENT_CREATE        5
#define RCCL_HIP_EVENT_DESTROY       6

#endif // RCCL_HIP_API_H
