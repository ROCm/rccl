/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "msccl/msccl_status.h"

mscclStatus& mscclGetStatus() {
  static mscclStatus status;
  return status;
}

mscclThreadLocalStatus& mscclGetThreadLocalStatus() {
  static thread_local mscclThreadLocalStatus threadLocalStatus;
  return threadLocalStatus;
}
