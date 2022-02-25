/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include <cstdlib>
namespace RcclUnitTesting
{
  TEST(AllReduce, Clique)
  {
    // Set clique env var prior to TestBed
    setenv("RCCL_ENABLE_CLIQUE", "1", 1);

    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes      = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes      = testBed.GetAllSupportedDataTypes();
    std::vector<ncclRedOp_t>    const redOps         = testBed.GetAllSupportedRedOps();
    std::vector<int>            const roots          = {0};
    std::vector<int>            const numElements    = {1048576, 53327, 1024};
    std::vector<bool>           const inPlaceList    = {false, true};
    std::vector<bool>           const managedMemList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements, inPlaceList, managedMemList);
    testBed.Finalize();

    unsetenv("RCCL_ENABLE_CLIQUE");
  }
}
