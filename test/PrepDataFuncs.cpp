/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "CollectiveArgs.hpp"
#include "PrepDataFuncs.hpp"
#include <cstdio>
#include <hip/hip_runtime.h>

namespace RcclUnitTesting
{
  ErrCode DefaultPrepareDataFunc(CollectiveArgs &collArgs)
  {
    switch (collArgs.funcType)
    {
    case ncclCollBroadcast:     return DefaultPrepData_Broadcast(collArgs);
    case ncclCollReduce:        return DefaultPrepData_Reduce(collArgs);
    case ncclCollAllGather:     return DefaultPrepData_AllGather(collArgs);
    case ncclCollReduceScatter: return DefaultPrepData_ReduceScatter(collArgs);
    case ncclCollAllReduce:     return DefaultPrepData_AllReduce(collArgs);
    case ncclCollGather:        return DefaultPrepData_Gather(collArgs);
    case ncclCollScatter:       return DefaultPrepData_Scatter(collArgs);
    case ncclCollAllToAll:      return DefaultPrepData_AllToAll(collArgs);
      //case ncclCollSendRecv:      return DefaultPrepData_SendRecv(collArgs);
    default:
      printf("[ERROR] Unknown func type %d\n", collArgs.funcType);
      return TEST_FAIL;
    }
  }

  ErrCode DefaultPrepData_Broadcast(CollectiveArgs &collArgs)
  {
    size_t const numBytes = collArgs.numInputElementsAllocated * DataTypeToBytes(collArgs.dataType);

    // Clear output for all ranks (done before filling input in case of in-place)
    CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numBytes));

    // Only root needs input pattern
    if (collArgs.globalRank == collArgs.root)
      collArgs.inputGpu.FillPattern(collArgs.dataType,
                                    collArgs.numInputElementsAllocated,
                                    collArgs.globalRank, true);

    // Otherwise all other ranks expected output is the same as input of root
    return collArgs.expected.FillPattern(collArgs.dataType,
                                         collArgs.numInputElementsAllocated,
                                         collArgs.root,
                                         false);
  }

  ErrCode DefaultPrepData_Reduce(CollectiveArgs &collArgs)
  {
    size_t const numBytes = collArgs.numInputElementsAllocated * DataTypeToBytes(collArgs.dataType);

    // Clear output for all ranks (done before filling input in case of in-place)
    CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numBytes));

    // Loop over all ranks and reduce
    PtrUnion tempInput;
    tempInput.AllocateCpuMem(numBytes);

    PtrUnion result;
    if (collArgs.root == collArgs.globalRank)
    {
      result.Attach(collArgs.expected.ptr);
      memset(result.ptr, 0, numBytes);
    }
    for (int rank = 0; rank < collArgs.totalRanks; ++rank)
    {
      tempInput.FillPattern(collArgs.dataType, collArgs.numInputElementsAllocated, rank, false);

      if (rank == collArgs.globalRank)
      {
        if (hipMemcpy(collArgs.inputGpu.ptr, tempInput.ptr, numBytes, hipMemcpyHostToDevice) != hipSuccess)
        {
          fprintf(stderr, "[ERROR] hipMemcpy to input failed\n");
          tempInput.FreeCpuMem();
          return TEST_FAIL;
        }
      }

      if (collArgs.root == collArgs.globalRank)
      {
        for (size_t i = 0; i < collArgs.numInputElementsAllocated; ++i)
          result.Reduce(tempInput, collArgs.dataType, i, collArgs.redOp);
      }
    }
    tempInput.FreeCpuMem();
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_AllGather(CollectiveArgs &collArgs)
  {
    PtrUnion tempInput;
    PtrUnion result;

    size_t const numBytes = collArgs.numInputElementsAllocated * DataTypeToBytes(collArgs.dataType);
    tempInput.AllocateCpuMem(numBytes);
    result.Attach(collArgs.expected.ptr);

    for (int rank = 0; rank < collArgs.totalRanks; ++rank)
    {
      tempInput.FillPattern(collArgs.dataType, collArgs.numInputElementsAllocated, rank, false);
      if (rank == collArgs.globalRank)
      {
        if (hipMemcpy(collArgs.inputGpu.ptr, tempInput.ptr, numBytes, hipMemcpyHostToDevice) != hipSuccess)
        {
          fprintf(stderr, "[ERROR] hipMemcpy to input failed\n");
          tempInput.FreeCpuMem();
          return TEST_FAIL;
        }
      }
      memcpy(result.I1 + (rank * numBytes), tempInput.ptr, numBytes);
    }
    tempInput.FreeCpuMem();
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_ReduceScatter(CollectiveArgs &collArgs)
  {
    PtrUnion tempInput;
    PtrUnion tempResult;

    size_t const numInputBytes = collArgs.numInputElementsAllocated * DataTypeToBytes(collArgs.dataType);
    size_t const numOutputBytes = collArgs.numOutputElementsAllocated * DataTypeToBytes(collArgs.dataType);

    // Clear output for all ranks (done before filling input in case of in-place)
    CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numOutputBytes));

    tempInput.AllocateCpuMem(numInputBytes);
    tempResult.AllocateCpuMem(numInputBytes);
    tempResult.ClearCpuMem(numInputBytes);

    for (int rank = 0; rank < collArgs.totalRanks; ++rank)
    {
      tempInput.FillPattern(collArgs.dataType, collArgs.numInputElementsAllocated, rank, false);
      for (size_t i = 0; i < collArgs.numInputElementsAllocated; ++i)
        tempResult.Reduce(tempInput, collArgs.dataType, i, collArgs.redOp);

      if (rank == collArgs.globalRank)
      {
        if (hipMemcpy(collArgs.inputGpu.ptr, tempInput.ptr, numInputBytes, hipMemcpyHostToDevice) != hipSuccess)
        {
          fprintf(stderr, "[ERROR] hipMemcpy to input failed\n");
          tempInput.FreeCpuMem();
          return TEST_FAIL;
        }
      }
    }
    tempInput.FreeCpuMem();

    memcpy(collArgs.expected.I1 + collArgs.globalRank * numOutputBytes,
           tempResult.I1 + collArgs.globalRank * numOutputBytes,
           numOutputBytes);
    tempResult.FreeCpuMem();
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_AllReduce(CollectiveArgs &collArgs)
  {
    size_t const numBytes = collArgs.numInputElementsAllocated * DataTypeToBytes(collArgs.dataType);

    // Clear output for all ranks (done before filling input in case of in-place)
    CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numBytes));

    // Loop over all ranks and reduce
    PtrUnion tempInput;
    tempInput.AllocateCpuMem(numBytes);

    PtrUnion result;
    result.Attach(collArgs.expected.ptr);
    memset(result.ptr, 0, numBytes);
    for (int rank = 0; rank < collArgs.totalRanks; ++rank)
    {
      tempInput.FillPattern(collArgs.dataType, collArgs.numInputElementsAllocated, rank, false);

      if (rank == collArgs.globalRank)
      {
        if (hipMemcpy(collArgs.inputGpu.ptr, tempInput.ptr, numBytes, hipMemcpyHostToDevice) != hipSuccess)
        {
          fprintf(stderr, "[ERROR] hipMemcpy to input failed\n");
          tempInput.FreeCpuMem();
          return TEST_FAIL;
        }
      }

      for (size_t i = 0; i < collArgs.numInputElementsAllocated; ++i)
        result.Reduce(tempInput, collArgs.dataType, i, collArgs.redOp);
    }
    tempInput.FreeCpuMem();
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_Gather(CollectiveArgs &collArgs)
  {
    PtrUnion tempInput;
    PtrUnion result;

    size_t const numInputBytes = collArgs.numInputElementsAllocated * DataTypeToBytes(collArgs.dataType);
    tempInput.AllocateCpuMem(numInputBytes);
    result.Attach(collArgs.expected.ptr);

    for (int rank = 0; rank < collArgs.totalRanks; ++rank)
    {
      tempInput.FillPattern(collArgs.dataType, collArgs.numInputElementsAllocated, rank, false);
      if (rank == collArgs.globalRank)
      {
        if (hipMemcpy(collArgs.inputGpu.ptr, tempInput.ptr, numInputBytes, hipMemcpyHostToDevice) != hipSuccess)
        {
          fprintf(stderr, "[ERROR] hipMemcpy to input failed\n");
          tempInput.FreeCpuMem();
          return TEST_FAIL;
        }
      }
      memcpy(result.I1 + (rank * numInputBytes), tempInput.ptr, numInputBytes);
    }
    tempInput.FreeCpuMem();
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_Scatter(CollectiveArgs &collArgs)
  {
    size_t const numInputBytes = collArgs.numInputElementsAllocated * DataTypeToBytes(collArgs.dataType);
    size_t const numOutputBytes = collArgs.numOutputElementsAllocated * DataTypeToBytes(collArgs.dataType);

    // Clear outputs on all ranks (prior to input in case of in-place)
    collArgs.outputGpu.ClearGpuMem(numOutputBytes);

    // Generate input on root rank - each rank will receive a portion
    PtrUnion tempInput;
    tempInput.AllocateCpuMem(numInputBytes);
    tempInput.FillPattern(collArgs.dataType, collArgs.numInputElementsAllocated, collArgs.root, false);

    // Copy input to root rank
    if (collArgs.globalRank == collArgs.root)
    {
      if (hipMemcpy(collArgs.inputGpu.ptr, tempInput.ptr, numInputBytes, hipMemcpyHostToDevice) != hipSuccess)
      {
        fprintf(stderr, "[ERROR] hipMemcpy to input failed\n");
        tempInput.FreeCpuMem();
        return TEST_FAIL;
      }
    }
    else
    {
      collArgs.inputGpu.ClearGpuMem(numInputBytes);
    }

    // Each rank receive a portion of the input
    memcpy(collArgs.expected.U1, tempInput.U1 + (collArgs.globalRank * numOutputBytes), numOutputBytes);

    tempInput.FreeCpuMem();
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_AllToAll(CollectiveArgs &collArgs)
  {
    size_t const numInputBytes = collArgs.numInputElementsAllocated * DataTypeToBytes(collArgs.dataType);
    size_t const numOutputBytes = collArgs.numOutputElementsAllocated * DataTypeToBytes(collArgs.dataType);
    size_t const numBytes = numInputBytes / collArgs.totalRanks;

    // Clear outputs on all ranks (prior to input in case of in-place)
    collArgs.outputGpu.ClearGpuMem(numOutputBytes);

    // Generate input on root rank - each rank will receive a portion
    PtrUnion tempInput;
    tempInput.AllocateCpuMem(numInputBytes);


    for (int rank = 0; rank < collArgs.totalRanks; ++rank)
    {
      tempInput.FillPattern(collArgs.dataType, collArgs.numInputElementsAllocated, rank, false);

      // Copy input
      if (rank == collArgs.globalRank)
      {
        if (hipMemcpy(collArgs.inputGpu.ptr, tempInput.ptr, numInputBytes, hipMemcpyHostToDevice) != hipSuccess)
        {
          fprintf(stderr, "[ERROR] hipMemcpy to input failed\n");
          tempInput.FreeCpuMem();
          return TEST_FAIL;
        }
      }
      memcpy(collArgs.expected.U1 + (numBytes * rank), tempInput.U1 + (numBytes * rank), numBytes);
    }
    tempInput.FreeCpuMem();
    return TEST_SUCCESS;
  }
}
