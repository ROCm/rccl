/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rcclBroadCastKernels.h"
#include "rcclLog.h"

template<typename DataType, typename VectorType>
rcclResult_t rcclInternalBcastRoot(DeviceControl_t *currTrack, size_t count, hipStream_t stream) {
    hipEvent_t event;
    hipEventCreateWithFlags(&event, hipEventReleaseToSystem);
    uint32_t i = 0;
    size_t numIter = count / (CHUNK_DWORDx4 * (sizeof(VectorType) / sizeof(DataType)));
    size_t offset = count - numIter * (CHUNK_DWORDx4 * (sizeof(VectorType) / sizeof(DataType)));
    for(;i<numIter;i++) {
        hipLaunchKernelGGL(CopyRoot, dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, i+1);
        hipEventRecord(event, stream);
        hipLaunchKernelGGL(rcclDoPeerChunk, dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, i+1);
    }
    if(offset != 0) {
        hipLaunchKernelGGL((CopyRootCnt<DataType, VectorType>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, i+1, offset);
        hipEventRecord(event, stream);
        hipLaunchKernelGGL(rcclDoPeerChunk, dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, i+1);
    }
    return rcclSuccess;
}

template<typename DataType, typename VectorType>
rcclResult_t rcclInternalBcast(DeviceControl_t *currTrack, size_t count, hipStream_t stream) {
    hipEvent_t event;
    hipEventCreateWithFlags(&event, hipEventReleaseToSystem);
    uint32_t i = 0;
    size_t numIter = count / (CHUNK_DWORDx4 * (sizeof(VectorType) / sizeof(DataType)));
    size_t offset = count - numIter * (CHUNK_DWORDx4 * (sizeof(VectorType) / sizeof(DataType)));
    for(;i<numIter;i++) {
        hipLaunchKernelGGL(rcclWaitForChunk, dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, i+1);
        hipLaunchKernelGGL(Copy, dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, i+1);
        hipEventRecord(event, stream);
        hipLaunchKernelGGL(rcclDoPeerChunk, dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, i+1);
    }
    if(offset != 0) {
        hipLaunchKernelGGL(rcclDoPeerChunk, dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, i+1);
        hipLaunchKernelGGL((CopyRootCnt<DataType, VectorType>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, i+1, offset);
        hipEventRecord(event, stream);
    }
    return rcclSuccess;
}
