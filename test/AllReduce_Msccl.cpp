/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/
#include <cstdlib>

#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(AllReduce, MscclAllPairs)
  {
    bool mscclEnabled = false;
    const char* mscclEnabledStr = getenv("RCCL_MSCCL_ENABLED");
    if (mscclEnabledStr == nullptr || strcmp(mscclEnabledStr, "1") != 0) {
      return;
    }

    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes      = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes      = {ncclInt8, ncclInt32, ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps         = {ncclSum, ncclProd};
    std::vector<int>            const roots          = {0};
    std::vector<int>            const numElements    = {1048576, 32768, 1024};
    std::vector<bool>           const inPlaceList    = {true};
    std::vector<bool>           const managedMemList = {true, false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements, inPlaceList, managedMemList);
    testBed.Finalize();
  }
}
