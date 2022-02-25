/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(Scatter, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes      = {ncclCollScatter};
    std::vector<ncclDataType_t> const dataTypes      = {ncclUint8, ncclUint32, ncclUint64};
    std::vector<ncclRedOp_t>    const redOps         = {ncclSum};
    std::vector<int>            const roots          = {0};
    std::vector<int>            const numElements    = {1048576, 53327, 1024};
    std::vector<bool>           const inPlaceList    = {false};
    std::vector<bool>           const managedMemList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements, inPlaceList, managedMemList);
    testBed.Finalize();
  }
}
