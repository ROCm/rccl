/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "CollectiveArgs.hpp"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{
  ErrCode CollectiveArgs::SetArgs(int             const  globalRank,
                                  int             const  totalRanks,
                                  int             const  deviceId,
                                  ncclFunc_t      const  funcType,
                                  ncclDataType_t  const  dataType,
                                  size_t          const  numInputElements,
                                  size_t          const  numOutputElements,
                                  ScalarTransport const  scalarTransport,
                                  OptionalColArgs const  &optionalColArgs)
  {
    // Free scalar based on previous scalarMode
    if (optionalColArgs.scalarMode != -1)
    {
      if (this->optionalArgs.localScalar.ptr != nullptr)
      {
        if (this->optionalArgs.scalarMode == 0) this->optionalArgs.localScalar.FreeGpuMem();
        if (this->optionalArgs.scalarMode == 1) hipHostFree(this->optionalArgs.localScalar.ptr);
      }
    }

    this->globalRank        = globalRank;
    this->totalRanks        = totalRanks;
    this->deviceId          = deviceId;
    this->funcType          = funcType;
    this->dataType          = dataType;
    this->numInputElements  = numInputElements;
    this->numOutputElements = numOutputElements;
    this->scalarTransport   = scalarTransport;
    this->optionalArgs      = optionalColArgs;

    if (this->optionalArgs.scalarMode != -1)
    {
      size_t const numBytes = DataTypeToBytes(dataType);
      if (this->optionalArgs.scalarMode == ncclScalarDevice)
      {
        CHECK_CALL(this->optionalArgs.localScalar.AllocateGpuMem(numBytes));
        CHECK_HIP(hipMemcpy(this->optionalArgs.localScalar.ptr, scalarTransport.ptr + (globalRank * numBytes),
                            numBytes, hipMemcpyHostToDevice));
      }
      else if (this->optionalArgs.scalarMode == ncclScalarHostImmediate)
      {
        CHECK_HIP(hipHostMalloc(&this->optionalArgs.localScalar.ptr, numBytes, 0));
        memcpy(this->optionalArgs.localScalar.ptr, scalarTransport.ptr + (globalRank * numBytes), numBytes);
      }
    }
    return TEST_SUCCESS;
  }

  ErrCode CollectiveArgs::AllocateMem(bool   const inPlace,
                                      bool   const useManagedMem)
  {
    this->numInputBytesAllocated     = this->numInputElements * DataTypeToBytes(this->dataType);
    this->numOutputBytesAllocated    = this->numOutputElements * DataTypeToBytes(this->dataType);
    this->numInputElementsAllocated  = this->numInputElements;
    this->numOutputElementsAllocated = this->numOutputElements;
    this->inPlace                    = inPlace;
    this->useManagedMem              = useManagedMem;

    if (hipSetDevice(this->deviceId) != hipSuccess)
    {
      ERROR("Unable to call hipSetDevice to set to GPU %d\n", this->deviceId);
      return TEST_FAIL;
    }

    if (inPlace)
    {
      if (this->funcType == ncclCollScatter)
      {
        CHECK_CALL(this->inputGpu.AllocateGpuMem(this->numInputBytesAllocated, useManagedMem));
        this->outputGpu.Attach(this->inputGpu.U1 + (this->globalRank  * this->numOutputBytesAllocated));
      }
      else if (this->funcType == ncclCollGather)
      {
        CHECK_CALL(this->outputGpu.AllocateGpuMem(this->numOutputBytesAllocated, useManagedMem));
        this->inputGpu.Attach(this->outputGpu.U1 + (this->globalRank * this->numInputBytesAllocated));
      }
      else
      {
        size_t const numBytes = std::max(this->numInputBytesAllocated, this->numOutputBytesAllocated);
        CHECK_CALL(this->inputGpu.AllocateGpuMem(numBytes, useManagedMem));
        this->outputGpu.Attach(this->inputGpu.ptr);
      }
      CHECK_CALL(this->expected.AllocateCpuMem(this->numOutputBytesAllocated));
    }
    else
    {
      CHECK_CALL(this->inputGpu.AllocateGpuMem(this->numInputBytesAllocated, useManagedMem));
      CHECK_CALL(this->outputGpu.AllocateGpuMem(this->numOutputBytesAllocated, useManagedMem));
      CHECK_CALL(this->expected.AllocateCpuMem(this->numOutputBytesAllocated));
    }
    CHECK_CALL(this->outputCpu.AllocateCpuMem(this->numOutputBytesAllocated));
    return TEST_SUCCESS;
  }

  ErrCode CollectiveArgs::PrepareData(CollFuncPtr const prepareDataFunc)
  {
    CollFuncPtr prepFunc = (prepareDataFunc == nullptr ? DefaultPrepareDataFunc : prepareDataFunc);
    return prepFunc(*this);
  }

  ErrCode CollectiveArgs::ValidateResults()
  {
    // Ignore non-root outputs for collectives with a root
    if (CollectiveArgs::UsesRoot(this->funcType) && this->optionalArgs.root != this->globalRank) return TEST_SUCCESS;

    size_t const numOutputBytes = (this->numOutputElements * DataTypeToBytes(this->dataType));

    CHECK_HIP(hipMemcpy(this->outputCpu.ptr, this->outputGpu.ptr, numOutputBytes, hipMemcpyDeviceToHost));

    bool isMatch = true;
    CHECK_CALL(this->outputCpu.IsEqual(this->dataType,
                                       this->numOutputElements,
                                       this->expected,
                                       true,
                                       isMatch));
    if (!isMatch) ERROR("Mismatch for %s\n", this->GetDescription().c_str());
    return isMatch ? TEST_SUCCESS : TEST_FAIL;
  }

  ErrCode CollectiveArgs::DeallocateMem()
  {
    // If in-place, either only inputGpu or outputGpu was allocated
    if (this->inPlace)
    {
      if (this->funcType == ncclCollGather)
        this->outputGpu.FreeGpuMem();
      else
        this->inputGpu.FreeGpuMem();
    }
    else
    {
      this->inputGpu.FreeGpuMem();
      this->outputGpu.FreeGpuMem();
    }

    this->outputCpu.FreeCpuMem();
    this->expected.FreeCpuMem();

    if (this->optionalArgs.localScalar.ptr != nullptr)
    {
      if (this->optionalArgs.scalarMode == 0) this->optionalArgs.localScalar.FreeGpuMem();
      if (this->optionalArgs.scalarMode == 1) CHECK_HIP(hipHostFree(this->optionalArgs.localScalar.ptr));
    }
    return TEST_SUCCESS;
  }

  std::string CollectiveArgs::GetDescription() const
  {
    std::stringstream ss;

    ss << "(Rank " << this->globalRank << ") ";
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
    case ncclCollAllToAllv:     ss << "ncclAllToAllv";     break;
    case ncclCollSend:          ss << "ncclSend";          break;
    case ncclCollRecv:          ss << "ncclRecv";          break;
    default:                    ss << "[Unknown]";         break;
    }

    ss << " " << ncclDataTypeNames[this->dataType] << " ";
    if (this->funcType == ncclCollReduce ||
        this->funcType == ncclCollReduceScatter ||
        this->funcType == ncclCollAllReduce)
    {
      if (this->optionalArgs.redOp < ncclNumOps)
      {
        ss << ncclRedOpNames[this->optionalArgs.redOp] << " ";
      }
      else
      {
        ss << "CustomScalar ";
        PtrUnion scalarsPerRank;
        scalarsPerRank.Attach(scalarsPerRank.ptr);
        switch (this->dataType)
        {
        case ncclInt8:     ss << scalarsPerRank.I1[this->globalRank]; break;
        case ncclUint8:    ss << scalarsPerRank.U1[this->globalRank]; break;
        case ncclInt32:    ss << scalarsPerRank.I4[this->globalRank]; break;
        case ncclUint32:   ss << scalarsPerRank.U4[this->globalRank]; break;
        case ncclInt64:    ss << scalarsPerRank.I8[this->globalRank]; break;
        case ncclUint64:   ss << scalarsPerRank.U8[this->globalRank]; break;
        case ncclFloat32:  ss << scalarsPerRank.F4[this->globalRank]; break;
        case ncclFloat64:  ss << scalarsPerRank.F8[this->globalRank]; break;
        case ncclBfloat16: ss << scalarsPerRank.B2[this->globalRank]; break;
        default:           ss << "(UNKNOWN)";
        }
        ss << " ";
      }
    }

    if (this->funcType == ncclCollBroadcast ||
        this->funcType == ncclCollReduce ||
        this->funcType == ncclCollGather ||
        this->funcType == ncclCollScatter)
    {
      ss << "Root " << this->optionalArgs.root << " ";
    }

    if (this->funcType == ncclCollSend ||
        this->funcType == ncclCollRecv)
    {
      ss << "Peer " << this->optionalArgs.root << " ";
    }

    ss << "#In: " << this->numInputElements;
    ss << " #Out: " << this->numOutputElements;

    return ss.str();
  }

  void CollectiveArgs::GetNumElementsForFuncType(ncclFunc_t const funcType,
                                                 int        const N,
                                                 int        const totalRanks,
                                                 int*             numInputElements,
                                                 int*             numOutputElements)
  {
    switch (funcType)
    {
    case ncclCollBroadcast:
    case ncclCollReduce:
    case ncclCollAllReduce:
      *numInputElements  = N;
      *numOutputElements = N;
      break;
    case ncclCollGather:
    case ncclCollAllGather:
      *numInputElements  = N;
      *numOutputElements = totalRanks * N;
      break;
    case ncclCollScatter:
    case ncclCollReduceScatter:
      *numInputElements  = totalRanks * N;
      *numOutputElements = N;
      break;
    case ncclCollAllToAll:
      *numInputElements = totalRanks * N;
      *numOutputElements = totalRanks * N;
      break;
    default:
      *numInputElements = N;
      *numOutputElements = N;
      break;
    }
  }

  bool CollectiveArgs::UsesReduce(ncclFunc_t const funcType)
  {
    return (funcType == ncclCollReduce    ||
            funcType == ncclCollAllReduce ||
            funcType == ncclCollReduceScatter);
  }

  bool CollectiveArgs::UsesRoot(ncclFunc_t const funcType)
  {
    return (funcType == ncclCollBroadcast ||
            funcType == ncclCollReduce    ||
            funcType == ncclCollGather    ||
            funcType == ncclCollScatter   ||
            funcType == ncclCollSend); // this is incorrect but it works because in Send root is not root it is the peer
  }
}
