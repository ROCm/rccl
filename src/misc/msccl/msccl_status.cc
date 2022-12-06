/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "msccl/msccl_status.h"

mscclStatus& mscclGetStatus() {
  static mscclStatus status;
  return status;
}
