/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "PtrUnion.hpp"

namespace RcclUnitTesting
{
  size_t DataTypeToBytes(ncclDataType_t const dataType)
  {
    switch (dataType)
    {
    case ncclInt8:   return 1;
    case ncclUint8:  return 1;
    case ncclInt32:  return 4;
    case ncclUint32: return 4;
    case ncclInt64:  return 8;
    case ncclUint64: return 8;
    case ncclFloat16: return 2;
    case ncclFloat32: return 4;
    case ncclFloat64: return 8;
    case ncclBfloat16: return 2;
    default:
      fprintf(stderr, "[ERROR] Unsupported datatype (%d)\n", dataType);
      exit(0);
    }
  }

  void PtrUnion::Attach(void *ptr)
  {
    this->ptr = ptr;
  }

  ErrCode PtrUnion::AllocateGpuMem(size_t const numBytes, bool const useManagedMem)
  {
    if (numBytes)
    {
      if (useManagedMem)
      {
        if (hipMallocManaged(&I1, numBytes) != hipSuccess)
        {
          fprintf(stderr, "[ERROR] Unable to allocate managed memory of GPU memory (%lu bytes)\n", numBytes);
          return TEST_FAIL;
        }
      }
      else
      {
        if (hipMalloc(&I1, numBytes) != hipSuccess)
        {
          fprintf(stderr, "[ERROR] Unable to allocate memory of GPU memory (%lu bytes)\n", numBytes);
          return TEST_FAIL;
        }
      }
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::AllocateCpuMem(size_t const numBytes)
  {
    if (numBytes)
    {
      this->ptr = calloc(numBytes, 1);
      if (!ptr)
      {
        fprintf(stderr, "[ERROR] Unable to allocate memory (%lu bytes)\n", numBytes);
        return TEST_FAIL;
      }
    }
    return TEST_SUCCESS;
  }

  void PtrUnion::FreeGpuMem()
  {
    if (this->ptr != nullptr)
    {
      hipFree(this->ptr);
      this->ptr = nullptr;
    }
  }

  void PtrUnion::FreeCpuMem()
  {
    if (this->ptr != nullptr)
    {
      free(this->ptr);
      this->ptr = nullptr;
    }
  }

  ErrCode PtrUnion::ClearGpuMem(size_t const numBytes)
  {
    if (hipMemset(this->ptr, 0, numBytes) != hipSuccess)
    {
      fprintf(stderr, "[ERROR] Unable to call hipMemset\n");
      return TEST_FAIL;
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::ClearCpuMem(size_t const numBytes)
  {
    memset(this->ptr, 0, numBytes);
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::FillPattern(ncclDataType_t const dataType,
                                size_t         const numElementsAllocated,
                                int            const globalRank,
                                bool           const isGpuMem)
  {
    PtrUnion temp;
    size_t const numBytes = numElementsAllocated * DataTypeToBytes(dataType);

    // If this is GPU memory, create a CPU temp buffer otherwise fill CPU memory directly
    if (isGpuMem)
      temp.AllocateCpuMem(numBytes);
    else
      temp.Attach(this->ptr);

    for (int i = 0; i < numElementsAllocated; i++)
    {
      int    valueI = (globalRank + i) % 256;
      double valueF = 1.0L/((double)valueI+1.0L);
      temp.Set(dataType, i, valueI, valueF);
    }

    // If this is GPU memory, copy from CPU temp buffer
    if (isGpuMem)
    {
      if (hipMemcpy(this->ptr, temp.ptr, numBytes, hipMemcpyHostToDevice) != hipSuccess)
      {
        printf("[ERROR] Unable to fill input with pattern for rank %d\n", globalRank);
        return TEST_FAIL;
      }
      temp.FreeCpuMem();
    }

    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::Set(ncclDataType_t const dataType, int const idx, int valueI, double valueF)
  {
    switch (dataType)
    {
    case ncclInt8:     I1[idx] = valueI; break;
    case ncclUint8:    U1[idx] = valueI; break;
    case ncclInt32:    I4[idx] = valueI; break;
    case ncclUint32:   U4[idx] = valueI; break;
    case ncclInt64:    I8[idx] = valueI; break;
    case ncclUint64:   U8[idx] = valueI; break;
    case ncclFloat32:  F4[idx] = valueF; break;
    case ncclFloat64:  F8[idx] = valueF; break;
    case ncclBfloat16: B2[idx] = rccl_bfloat16(static_cast<float>(valueF)); break;
    default:
      fprintf(stderr, "[ERROR] Unsupported datatype\n");
      return TEST_FAIL;
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::Get(ncclDataType_t const dataType, int const idx, int& valueI, double& valueF)
  {
    switch (dataType)
    {
    case ncclInt8:     valueI = I1[idx]; break;
    case ncclUint8:    valueI = I1[idx]; break;
    case ncclInt32:    valueI = I4[idx]; break;
    case ncclUint32:   valueI = U4[idx]; break;
    case ncclInt64:    valueI = I8[idx]; break;
    case ncclUint64:   valueI = U8[idx]; break;
    case ncclFloat32:  valueF = F4[idx]; break;
    case ncclFloat64:  valueF = F8[idx]; break;
    case ncclBfloat16: valueF = B2[idx]; break;
    default:
      fprintf(stderr, "[ERROR] Unsupported datatype\n");
      return TEST_FAIL;
    }
    return TEST_SUCCESS;
  }

  bool PtrUnion::IsEqual(PtrUnion const& expected, ncclDataType_t const dataType, int const idx, bool const verbose)
  {
    bool isMatch = true;
    switch (dataType)
    {
    case ncclInt8:    isMatch = (I1[idx] == expected.I1[idx]); break;
    case ncclUint8:   isMatch = (U1[idx] == expected.U1[idx]); break;
    case ncclInt32:   isMatch = (I4[idx] == expected.I4[idx]); break;
    case ncclUint32:  isMatch = (U4[idx] == expected.U4[idx]); break;
    case ncclInt64:   isMatch = (I8[idx] == expected.I8[idx]); break;
    case ncclUint64:  isMatch = (U8[idx] == expected.U8[idx]); break;
    case ncclFloat32: isMatch = (fabs(F4[idx] - expected.F4[idx]) < 1e-5); break;
    case ncclFloat64: isMatch = (fabs(F8[idx] - expected.F8[idx]) < 1e-12); break;
    case ncclBfloat16: isMatch = (fabs((float)B2[idx] - (float)expected.B2[idx]) < 9e-2); break;
    default:
      fprintf(stderr, "[ERROR] Unsupported datatype\n");
      exit(0);
    }

    if (verbose && !isMatch)
    {
      switch (dataType)
      {
      case ncclInt8:
        printf("Expected %d.  Actual %d at index %d\n", expected.I1[idx], I1[idx], idx); break;
      case ncclUint8:
        printf("Expected %u.  Actual %u at index %d\n", expected.U1[idx], U1[idx], idx); break;
      case ncclInt32:
        printf("Expected %d.  Actual %d at index %d\n", expected.I4[idx], I4[idx], idx); break;
      case ncclUint32:
        printf("Expected %u.  Actual %u at index %d\n", expected.U4[idx], U4[idx], idx); break;
      case ncclInt64:
        printf("Expected %ld.  Actual %ld at index %d\n", expected.I8[idx], I8[idx], idx); break;
      case ncclUint64:
        printf("Expected %lu.  Actual %lu at index %d\n", expected.U8[idx], U8[idx], idx); break;
      case ncclFloat32:
        printf("Expected %f.  Actual %f at index %d\n", expected.F4[idx], F4[idx], idx); break;
      case ncclFloat64:
        printf("Expected %lf.  Actual %lf at index %d\n", expected.F8[idx], F8[idx], idx); break;
      case ncclBfloat16:
        printf("Expected %f.  Actual %f at index %d\n", (float)expected.B2[idx], (float)B2[idx], idx); break;
      default:
        fprintf(stderr, "[ERROR] Unsupported datatype\n");
        exit(0);
      }
    }
    return isMatch;
  }

  void PtrUnion::Reduce(PtrUnion const& input, ncclDataType_t const dataType, size_t const idx, ncclRedOp_t const op)
  {
    switch (dataType)
    {
    case ncclInt8:     I1[idx] = ReduceOp(op, I1[idx], input.I1[idx]); break;
    case ncclUint8:    U1[idx] = ReduceOp(op, U1[idx], input.U1[idx]); break;
    case ncclInt32:    I4[idx] = ReduceOp(op, I4[idx], input.I4[idx]); break;
    case ncclUint32:   U4[idx] = ReduceOp(op, U4[idx], input.U4[idx]); break;
    case ncclInt64:    I8[idx] = ReduceOp(op, I8[idx], input.I8[idx]); break;
    case ncclUint64:   U8[idx] = ReduceOp(op, U8[idx], input.U8[idx]); break;
    case ncclFloat32:  F4[idx] = ReduceOp(op, F4[idx], input.F4[idx]); break;
    case ncclFloat64:  F8[idx] = ReduceOp(op, F8[idx], input.F8[idx]); break;
    case ncclBfloat16: B2[idx] = ReduceOp(op, B2[idx], input.B2[idx]); break;
    default:
      fprintf(stderr, "[ERROR] Unsupported datatype\n");
      exit(0);
    }
  }

  void PtrUnion::Divide(ncclDataType_t const dataType, size_t const idx, int const divisor)
  {
    switch (dataType)
    {
    case ncclInt8:     I1[idx] /= divisor; break;
    case ncclUint8:    U1[idx] /= divisor; break;
    case ncclInt32:    I4[idx] /= divisor; break;
    case ncclUint32:   U4[idx] /= divisor; break;
    case ncclInt64:    I8[idx] /= divisor; break;
    case ncclUint64:   U8[idx] /= divisor; break;
    case ncclFloat32:  F4[idx] /= divisor; break;
    case ncclFloat64:  F8[idx] /= divisor; break;
    case ncclBfloat16: B2[idx] = (rccl_bfloat16((float)(B2[idx]) / divisor)); break;
    default:
      fprintf(stderr, "[ERROR] Unsupported datatype\n");
      exit(0);
    }
  }
}
