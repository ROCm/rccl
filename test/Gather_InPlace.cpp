/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(Gather, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes      = {ncclCollGather};
    std::vector<ncclDataType_t> const dataTypes      = {ncclInt8, ncclInt32, ncclInt64, ncclFloat16};
    std::vector<ncclRedOp_t>    const redOps         = {ncclSum};
    std::vector<int>            const roots          = {0};
    std::vector<int>            const numElements    = {1048576, 53327, 1024};
    std::vector<bool>           const inPlaceList    = {true};
    std::vector<bool>           const managedMemList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements, inPlaceList, managedMemList);
    testBed.Finalize();
  }
}
