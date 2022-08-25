/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(ReduceScatter, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes      = {ncclCollReduceScatter};
    std::vector<ncclDataType_t> const dataTypes      = {ncclFloat16, ncclFloat32, ncclFloat64, ncclBfloat16};
    std::vector<ncclRedOp_t>    const redOps         = {ncclMin, ncclMax, ncclAvg};
    std::vector<int>            const roots          = {0};
    std::vector<int>            const numElements    = {1048576, 53327, 5461, 1024};
    std::vector<bool>           const inPlaceList    = {false};
    std::vector<bool>           const managedMemList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements, inPlaceList, managedMemList);
    testBed.Finalize();
  }
}
