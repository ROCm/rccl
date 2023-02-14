/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_KERNEL_H_
#define MSCCL_KERNEL_H_

#define MSCCL_KERNEL_ENTRY_NAME(devredop, type, proto) mscclKernel_##devredop##_##type##_##proto

#define MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE_PROTO(devredop, type, proto) \
__global__ void MSCCL_KERNEL_ENTRY_NAME(devredop, type, proto)(struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork* work);

#define MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, type) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE_PROTO(devredop, type, LL) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE_PROTO(devredop, type, LL128) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE_PROTO(devredop, type, Simple)

#define MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP(devredop) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int8_t) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint8_t) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int32_t) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint32_t) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int64_t) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint64_t) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, half) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, float) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, double) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, rccl_bfloat16)

#define MSCCL_DECL_KERNEL_ENTRY_FUNC() \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP(Sum) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP(Prod) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP(Max) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP(Min)

MSCCL_DECL_KERNEL_ENTRY_FUNC()

#endif
