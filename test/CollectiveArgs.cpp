/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "CollectiveArgs.hpp"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{
  ErrCode CollectiveArgs::SetArgs(int            const globalRank,
                                  int            const totalRanks,
                                  int            const deviceId,
                                  ncclFunc_t     const funcType,
                                  ncclDataType_t const dataType,
                                  ncclRedOp_t    const redOp,
                                  int            const root,
                                  size_t         const numInputElements,
                                  size_t         const numOutputElements)
  {
    this->globalRank        = globalRank;
    this->totalRanks        = totalRanks;
    this->deviceId          = deviceId;
    this->funcType          = funcType;
    this->dataType          = dataType;
    this->redOp             = redOp;
    this->root              = root;
    this->numInputElements  = numInputElements;
    this->numOutputElements = numOutputElements;
    return TEST_SUCCESS;
  }

  ErrCode CollectiveArgs::AllocateMem(size_t const numInputBytesToAllocate,
                                      size_t const numOutputBytesToAllocate,
                                      bool   const inPlace,
                                      bool   const useManagedMem)
  {
    this->numInputBytesAllocated     = numInputBytesToAllocate;
    this->numOutputBytesAllocated    = numOutputBytesToAllocate;
    this->inPlace                    = inPlace;
    this->useManagedMem              = useManagedMem;
    this->numInputElementsAllocated  = numInputBytesToAllocate / DataTypeToBytes(this->dataType);
    this->numOutputElementsAllocated = numOutputBytesToAllocate / DataTypeToBytes(this->dataType);

    if (hipSetDevice(this->deviceId) != hipSuccess)
    {
      printf("[ERROR] Unnable to set device to %d\n", this->deviceId);
      return TEST_FAIL;
    }

    if (inPlace)
    {
      size_t const numBytes = std::max(numInputBytesToAllocate, numOutputBytesToAllocate);
      CHECK_CALL(this->inputGpu.AllocateGpuMem(numBytes, useManagedMem));
      this->outputGpu.Attach(this->inputGpu.ptr);
      CHECK_CALL(this->expected.AllocateCpuMem(numBytes));
    }
    else
    {
      CHECK_CALL(this->inputGpu.AllocateGpuMem(numInputBytesToAllocate, useManagedMem));
      CHECK_CALL(this->outputGpu.AllocateGpuMem(numOutputBytesToAllocate, useManagedMem));
      CHECK_CALL(this->expected.AllocateCpuMem(numOutputBytesToAllocate));
    }
    CHECK_CALL(this->outputCpu.AllocateCpuMem(numOutputBytesToAllocate));
    return TEST_SUCCESS;
  }

  ErrCode CollectiveArgs::PrepareData(CollFuncPtr const prepareDataFunc)
  {
    CollFuncPtr prepFunc = (prepareDataFunc == nullptr ? DefaultPrepareDataFunc : prepareDataFunc);
    return prepFunc(*this);
  }

  ErrCode CollectiveArgs::ValidateResults()
  {
    size_t const numOutputBytes = (this->numOutputElements * DataTypeToBytes(this->dataType));

    CHECK_HIP(hipMemcpy(this->outputCpu.ptr, this->outputGpu.ptr, numOutputBytes, hipMemcpyDeviceToHost));

    bool isMatch = true;
    for (size_t idx = 0; isMatch && idx < this->numOutputElements; ++idx)
    {
      isMatch &= this->outputCpu.IsEqual(this->expected,
                                         this->dataType,
                                         idx,
                                         true);
      if (!isMatch) printf("[ERROR]: %s\n", this->GetDescription().c_str());
    }
    return isMatch ? TEST_SUCCESS : TEST_FAIL;
  }

  ErrCode CollectiveArgs::DeallocateMem()
  {
    this->inputGpu.FreeGpuMem();
    this->outputGpu.FreeGpuMem();
    this->outputCpu.FreeCpuMem();
    this->expected.FreeCpuMem();
    return TEST_SUCCESS;
  }

  std::string CollectiveArgs::GetDescription() const
  {
    std::stringstream ss;

    ss << "(Rank " << this->globalRank << " of " << this->totalRanks << ") ";
    switch (this->funcType)
    {
    case ncclCollBroadcast:     ss << "ncclBroadcast";     break;
    case ncclCollReduce:        ss << "ncclReduce";        break;
    case ncclCollAllGather:     ss << "ncclAllGather";     break;
    case ncclCollReduceScatter: ss << "ncclReduceScatter"; break;
    case ncclCollAllReduce:     ss << "ncclAllReduce";     break;
    case ncclCollGather:        ss << "ncclGather";        break;
    case ncclCollScatter:       ss << "ncclScatter";       break;
    case ncclCollAllToAll:      ss << "ncclAllToAll";      break;
    case ncclCollSend:          ss << "ncclSend";          break;
    case ncclCollRecv:          ss << "ncclRevv";          break;
    default:                    ss << "[Unknown]";         break;
    }

    ss << ncclDataTypeNames[this->dataType] << " ";
    if (this->funcType == ncclCollReduce ||
        this->funcType == ncclCollReduceScatter ||
        this->funcType == ncclCollAllReduce)
    {
      ss << ncclRedOpNames[this->redOp] << " ";
    }

    if (this->funcType == ncclCollBroadcast ||
        this->funcType == ncclCollReduce ||
        this->funcType == ncclCollGather ||
        this->funcType == ncclCollScatter)
    {
      ss << "Root " << this->root << " ";
    }

    if (this->funcType == ncclCollSend ||
        this->funcType == ncclCollRecv)
    {
      ss << "Peer " << this->root << " ";
    }

    ss << "#In: " << this->numInputElements;
    ss << " #Out: " << this->numOutputElements;

    return ss.str();
  }
}
