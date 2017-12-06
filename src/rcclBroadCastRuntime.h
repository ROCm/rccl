/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rcclBroadCastKernels.h"
#include "rcclLog.h"

template<typename DataType, typename VectorType>
rcclResult_t rcclInternalBcastRoot(DeviceControl_t *currTrack, size_t numIter, size_t offsetCnt, hipStream_t stream) {
    hipEvent_t event;
    hipEventCreateWithFlags(&event, hipEventReleaseToSystem);
    int i = 0;
    for(;i<numIter;i++) {
        hipLaunchKernelGGL(CopyRoot, dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, i+1);
        hipEventRecord(event, stream);
        hipLaunchKernelGGL(rcclDoPeerChunk, dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, i+1);
    }
    if(offsetCnt != 0) {
        hipLaunchKernelGGL((CopyRootCnt<DataType, VectorType>), dim3(1,1,1), dim3(WI,1,1), 0, stream,currTrack, i+1, offsetCnt);
        hipEventRecord(event, stream);
    }
    return rcclSuccess;
}

template<typename DataType, typename VectorType>
rcclResult_t rcclInternalBcast(DeviceControl_t *currTrack, size_t numIter, size_t offsetCnt, hipStream_t stream) {
    int i = 0;
    hipEvent_t event;
    hipEventCreateWithFlags(&event, hipEventReleaseToSystem);
    for(;i<numIter;i++) {
        hipLaunchKernelGGL(rcclWaitForChunk, dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, i+1);
        hipLaunchKernelGGL(Copy, dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, i+1);
        hipEventRecord(event, stream);
        hipLaunchKernelGGL(rcclDoPeerChunk, dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, i+1);
    }
    if(offsetCnt != 0) {
        hipLaunchKernelGGL(rcclWaitForChunk, dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, i+1);
        hipLaunchKernelGGL((CopyCnt<DataType, VectorType>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, i+1, offsetCnt);
        hipEventRecord(event, stream);
    }
    return rcclSuccess;
}
