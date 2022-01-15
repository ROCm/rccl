/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once
#include "PtrUnion.hpp"
#include "PrepDataFuncs.hpp"
#include "rccl.h"

namespace RcclUnitTesting
{
  // Enumeration of all collective functions currently supported
  typedef enum
  {
    ncclCollBroadcast,
    ncclCollReduce,
    ncclCollAllGather,
    ncclCollReduceScatter,
    ncclCollAllReduce,
    ncclCollGather,
    ncclCollScatter,
    ncclCollAllToAll,
    ncclCollSend,
    ncclCollRecv
  } ncclFunc_t;

  char const ncclDataTypeNames[ncclNumTypes][32] =
  {
    "ncclInt8",
    "ncclUint8",
    "ncclInt32",
    "ncclUint32",
    "ncclInt64",
    "ncclUint64",
    "ncclFloat16",
    "ncclFloat32",
    "ncclFloat64",
    "ncclBfloat16"
  };

  char const ncclRedOpNames[ncclNumOps][32] =
  {
    "sum",
    "prod",
    "max",
    "min",
    "avg"
  };

  class CollectiveArgs;

  // Function pointer for functions that operate on CollectiveArgs
  // e.g. For filling input / computing expected results
  typedef ErrCode (*CollFuncPtr)(CollectiveArgs &);

  class CollectiveArgs
  {
  public:
    // Arguments to execute
    int            globalRank;
    int            totalRanks;
    int            deviceId;
    ncclFunc_t     funcType;
    ncclDataType_t dataType;
    ncclRedOp_t    redOp;
    int            root;    // Used as "peer" for Send/Recv
    size_t         numInputElements;
    size_t         numOutputElements;

    // Data
    PtrUnion       inputGpu;
    PtrUnion       outputGpu;
    PtrUnion       outputCpu;
    PtrUnion       expected;
    bool           inPlace;
    bool           useManagedMem;
    size_t         numInputBytesAllocated;
    size_t         numOutputBytesAllocated;
    size_t         numInputElementsAllocated;
    size_t         numOutputElementsAllocated;

    // Set collective arguments
    ErrCode SetArgs(int            const globalRank,
                    int            const totalRanks,
                    int            const deviceId,
                    ncclFunc_t     const funcType,
                    ncclDataType_t const dataType,
                    ncclRedOp_t    const redOp,
                    int            const root,
                    size_t         const numInputElements,
                    size_t         const numOutputElements);

    // Allocates GPU memory for input/output and CPU memory for expected
    // When inPlace is true, input and output share the same memory
    ErrCode AllocateMem(size_t const numInputBytesToAllocate,
                        size_t const numOutputBytesToAllocate,
                        bool   const inPlace,
                        bool   const useManagedMem);

    ErrCode PrepareData(CollFuncPtr const prepareDataFunc);

    ErrCode ValidateResults();

    ErrCode DeallocateMem();

    std::string GetDescription() const;
  };
}
