/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rcclAllReduceKernels.h"

template<typename DataType, typename VectorType, rcclRedOp_t Op>
rcclResult_t rcclInternalAllReduce(DeviceControl_t *currTrack, int rank, int numGpus, size_t count, hipStream_t stream, hipEvent_t event) {
    size_t numIter = count / ((CHUNK_DWORDx4 * (sizeof(VectorType) / sizeof(DataType))) * numGpus);
    size_t offSet = count % ((CHUNK_DWORDx4 * (sizeof(VectorType) / sizeof(DataType))) * numGpus);
    VectorType *tmpSrc = reinterpret_cast<VectorType*>(currTrack->controlBuffer);
    VectorType *tmpDst = reinterpret_cast<VectorType*>(currTrack->nextPeer->controlBuffer);
    if(Op == rcclSum) {
        int currChunkId = 0;
        int loop = 0;
        if(numIter > 0) {
            for(int loop = 0;loop < numIter; loop++) {
                int i=0;
                int tmpId = rank%numGpus;
                int Id = tmpId + numGpus*loop;
                currChunkId++;
                hipLaunchKernelGGL((rcclAllReduceFirstCopy), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, tmpDst + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));

                hipEventRecord(event, stream);
                hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);

                if(numGpus > 2) {
                    for(i=2;i<numGpus;i++) {
                        tmpId = (tmpId + numGpus - 1)%numGpus;
                        Id = tmpId + numGpus*loop;
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        hipLaunchKernelGGL((rcclAllReduceOpCopy<DataType, VectorType, rcclSum>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, tmpDst + tmpId * CHUNK_DWORDx4, tmpSrc + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    }

                    tmpId = (tmpId + numGpus - 1)%numGpus;
                    Id = tmpId + numGpus*loop;
                    hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    currChunkId++;
                    hipLaunchKernelGGL((rcclAllReduceOpCopynextPeerDst<DataType, VectorType, rcclSum, true>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4, tmpSrc + tmpId * CHUNK_DWORDx4, Id * CHUNK_DWORDx4);
                    hipEventRecord(event, stream);
                    hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    i++;
                    for(int j=1; j<numGpus;j++,i++) {
                        tmpId = (tmpId + numGpus - 1)%numGpus;
                        Id = tmpId + numGpus*loop;
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        hipLaunchKernelGGL((rcclAllReduceCopynextPeerDst<VectorType>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4);
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    }
                } else if(numGpus == 2) {
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        tmpId = (rank+1)%numGpus;
                        Id = tmpId + numGpus * loop;
                        hipLaunchKernelGGL((rcclAllReduceOpCopynextPeerDst<DataType, VectorType, rcclSum, true>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4, tmpSrc + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        tmpId = (rank+2)%numGpus;
                        Id = tmpId + numGpus * loop;
                        hipLaunchKernelGGL((rcclAllReduceCopynextPeerDst<VectorType>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4);
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                }

            }
        }
        hipEventRecord(event, stream);

        if(offSet != 0) {
            size_t count = numIter * numGpus * CHUNK_DWORDx4*sizeof(VectorType)/sizeof(DataType);
            hipLaunchKernelGGL((rcclAllReduceOpCopyTail<DataType, VectorType, rcclSum>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, numGpus, count, offSet);
        }

    }

    if(Op == rcclProd) {
        int currChunkId = 0;

        int loop = 0;
        if(numIter > 0) {
            for(int loop = 0;loop < numIter; loop++) {
                int i=0;
                int tmpId = rank%numGpus;
                int Id = tmpId + numGpus*loop;
                currChunkId++;
                hipLaunchKernelGGL((rcclAllReduceFirstCopy), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, tmpDst + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));

                hipEventRecord(event, stream);
                hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);

                if(numGpus > 2) {
                    for(i=2;i<numGpus;i++) {
                        tmpId = (tmpId + numGpus - 1)%numGpus;
                        Id = tmpId + numGpus*loop;
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        hipLaunchKernelGGL((rcclAllReduceOpCopy<DataType, VectorType, rcclProd>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, tmpDst + tmpId * CHUNK_DWORDx4, tmpSrc + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    }

                    tmpId = (tmpId + numGpus - 1)%numGpus;
                    Id = tmpId + numGpus*loop;
                    hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    currChunkId++;
                    hipLaunchKernelGGL((rcclAllReduceOpCopynextPeerDst<DataType, VectorType, rcclProd, true>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4, tmpSrc + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));
                    hipEventRecord(event, stream);
                    hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    i++;
                    for(int j=1; j<numGpus;j++,i++) {
                        tmpId = (tmpId + numGpus - 1)%numGpus;
                        Id = tmpId + numGpus*loop;
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        hipLaunchKernelGGL((rcclAllReduceCopynextPeerDst<VectorType>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4);
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    }
                }
                else if(numGpus == 2) {
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        tmpId = (rank+1)%numGpus;
                        Id = tmpId + numGpus * loop;
                        hipLaunchKernelGGL((rcclAllReduceOpCopynextPeerDst<DataType, VectorType, rcclProd, true>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4, tmpSrc + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        tmpId = (rank+2)%numGpus;
                        Id = tmpId + numGpus * loop;
                        hipLaunchKernelGGL((rcclAllReduceCopynextPeerDst<VectorType>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4);
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                }
            }
        }

        if(offSet != 0) {
            hipLaunchKernelGGL((rcclAllReduceOpCopyTail<DataType, VectorType, rcclProd>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, numGpus, numIter*numGpus*CHUNK_DWORDx4*sizeof(VectorType)/sizeof(DataType), offSet);
        }
    }


    if(Op == rcclMax) {
        int currChunkId = 0;

        int loop = 0;
        if(numIter > 0) {
            for(int loop = 0;loop < numIter; loop++) {
                int i=0;
                int tmpId = rank%numGpus;
                int Id = tmpId + numGpus*loop;
                currChunkId++;
                hipLaunchKernelGGL((rcclAllReduceFirstCopy), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, tmpDst + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));

                hipEventRecord(event, stream);
                hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);

                if(numGpus > 2) {
                    for(i=2;i<numGpus;i++) {
                        tmpId = (tmpId + numGpus - 1)%numGpus;
                        Id = tmpId + numGpus*loop;
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        hipLaunchKernelGGL((rcclAllReduceOpCopy<DataType, VectorType, rcclMax>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, tmpDst + tmpId * CHUNK_DWORDx4, tmpSrc + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    }

                    tmpId = (tmpId + numGpus - 1)%numGpus;
                    Id = tmpId + numGpus*loop;
                    hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    currChunkId++;
                    hipLaunchKernelGGL((rcclAllReduceOpCopynextPeerDst<DataType, VectorType, rcclMax, true>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4, tmpSrc + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));
                    hipEventRecord(event, stream);
                    hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    i++;
                    for(int j=1; j<numGpus;j++,i++) {
                        tmpId = (tmpId + numGpus - 1)%numGpus;
                        Id = tmpId + numGpus*loop;
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        hipLaunchKernelGGL((rcclAllReduceCopynextPeerDst<VectorType>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4);
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    }
                } else if(numGpus == 2) {
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        tmpId = (rank+1)%numGpus;
                        Id = tmpId + numGpus * loop;
                        hipLaunchKernelGGL((rcclAllReduceOpCopynextPeerDst<DataType, VectorType, rcclProd, true>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4, tmpSrc + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        tmpId = (rank+2)%numGpus;
                        Id = tmpId + numGpus * loop;
                        hipLaunchKernelGGL((rcclAllReduceCopynextPeerDst<VectorType>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4);
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                }

            }
        }

        if(offSet != 0) {
            hipLaunchKernelGGL((rcclAllReduceOpCopyTail<DataType, VectorType, rcclMax>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, numGpus, numIter*numGpus*CHUNK_DWORDx4*sizeof(VectorType)/sizeof(DataType), offSet);
        }
    }

    if(Op == rcclMin) {
        int currChunkId = 0;

        int loop = 0;
        if(numIter > 0) {
            for(int loop = 0;loop < numIter; loop++) {
                int i=0;
                int tmpId = rank%numGpus;
                int Id = tmpId + numGpus*loop;
                currChunkId++;
                hipLaunchKernelGGL((rcclAllReduceFirstCopy), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, tmpDst + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));

                hipEventRecord(event, stream);
                hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);

                if(numGpus > 2) {
                    for(i=2;i<numGpus;i++) {
                        tmpId = (tmpId + numGpus - 1)%numGpus;
                        Id = tmpId + numGpus*loop;
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        hipLaunchKernelGGL((rcclAllReduceOpCopy<DataType, VectorType, rcclMin>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, tmpDst + tmpId * CHUNK_DWORDx4, tmpSrc + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    }

                    tmpId = (tmpId + numGpus - 1)%numGpus;
                    Id = tmpId + numGpus*loop;
                    hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    currChunkId++;
                    hipLaunchKernelGGL((rcclAllReduceOpCopynextPeerDst<DataType, VectorType, rcclMin, true>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4, tmpSrc + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));
                    hipEventRecord(event, stream);
                    hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    i++;
                    for(int j=1; j<numGpus;j++,i++) {
                        tmpId = (tmpId + numGpus - 1)%numGpus;
                        Id = tmpId + numGpus*loop;
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        hipLaunchKernelGGL((rcclAllReduceCopynextPeerDst<VectorType>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4);
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                    }
                } else if(numGpus == 2) {
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        tmpId = (rank+1)%numGpus;
                        Id = tmpId + numGpus * loop;
                        hipLaunchKernelGGL((rcclAllReduceOpCopynextPeerDst<DataType, VectorType, rcclMin, true>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4, tmpSrc + tmpId * CHUNK_DWORDx4, size_t(Id * CHUNK_DWORDx4));
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        hipLaunchKernelGGL((rcclWaitForChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                        tmpId = (rank+2)%numGpus;
                        Id = tmpId + numGpus * loop;
                        hipLaunchKernelGGL((rcclAllReduceCopynextPeerDst<VectorType>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, Id * CHUNK_DWORDx4);
                        hipEventRecord(event, stream);
                        currChunkId++;
                        hipLaunchKernelGGL((rcclDoPeerChunk), dim3(1,1,1), dim3(1,1,1), 0, stream, currTrack, currChunkId);
                }

            }
        }

        if(offSet != 0) {
            hipLaunchKernelGGL((rcclAllReduceOpCopyTail<DataType, VectorType, rcclMin>), dim3(1,1,1), dim3(WI,1,1), 0, stream, currTrack, numGpus, numIter*numGpus*CHUNK_DWORDx4*sizeof(VectorType)/sizeof(DataType), offSet);
        }
    }

    return rcclSuccess;
}
