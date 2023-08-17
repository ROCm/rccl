/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include "StandaloneUtils.hpp"

TEST(Standalone, SplitComms) {
    std::vector<ncclComm_t> comms;
    int numDevices;

    HIPCALL(hipGetDeviceCount(&numDevices));

    if (numDevices < 2) {
        GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    comms.resize(numDevices);
    std::vector<ncclComm_t> subComms(numDevices);

    //Initialize the original comms
    NCCLCHECK(ncclCommInitAll(comms.data(), numDevices, nullptr));

    /*==================== Rank check test ===================*/
    int numSubComms = 2;

    std::map<int, int> mapCounter;
    NCCLCHECK(ncclGroupStart());
    for (int localRank = 0; localRank < numDevices; localRank++) {
        NCCLCHECK(ncclCommSplit(comms[localRank], localRank % numSubComms, localRank, &subComms[localRank], NULL));
        mapCounter[localRank % numSubComms]++;
    }
    NCCLCHECK(ncclGroupEnd());

    for (int i = 0; i < numDevices; i++) {
        int subCommRank, subCommNRank;
        NCCLCHECK(ncclCommUserRank(subComms[i], &subCommRank));
        NCCLCHECK(ncclCommCount(subComms[i], &subCommNRank));

        // Assert sub-communication properties
        ASSERT_EQ(subCommRank, i / numSubComms);
        ASSERT_EQ(subCommNRank, mapCounter[i % numSubComms]);
    }

    // Destroy sub-comms
    for (auto& subComm : subComms) {
        NCCLCHECK(ncclCommDestroy(subComm));
    }

    /*================== One color test ====================*/
    NCCLCHECK(ncclGroupStart());
    for (int localRank = 0; localRank < numDevices; localRank++) {
        NCCLCHECK(ncclCommSplit(comms[localRank], 0, localRank, &subComms[localRank], NULL));
    }
    NCCLCHECK(ncclGroupEnd());

    // Validate results
    for (int i = 0; i < numDevices; i++) {
        int originalRank, originalNRank;
        NCCLCHECK(ncclCommUserRank(comms[i], &originalRank));
        NCCLCHECK(ncclCommCount(comms[i], &originalNRank));

        int subCommRank, subCommNRank;
        NCCLCHECK(ncclCommUserRank(subComms[i], &subCommRank));
        NCCLCHECK(ncclCommCount(subComms[i], &subCommNRank));
        
        ASSERT_EQ(originalRank, subCommRank);
        ASSERT_EQ(originalNRank, subCommNRank);
    }

    // Destroy sub-comms
    for (auto& subComm : subComms) {
        NCCLCHECK(ncclCommDestroy(subComm));
    }

    /*==================== Reduce ranks test =======================*/
    // Number of ranks to duplicate from the original comm
    int numReducedRanks = numDevices / 2;

    NCCLCHECK(ncclGroupStart());
    for (int localRank = 0; localRank < numDevices; localRank++) {
        NCCLCHECK(ncclCommSplit(comms[localRank], localRank < numReducedRanks ? 0 : NCCL_SPLIT_NOCOLOR, localRank, &subComms[localRank], NULL));
    }
    NCCLCHECK(ncclGroupEnd());

    // Validate results
    for (int i = 0; i < numDevices; i++) {
        int originalRank, originalNRank;
        NCCLCHECK(ncclCommUserRank(comms[i], &originalRank));
        NCCLCHECK(ncclCommCount(comms[i], &originalNRank));

        if (i < numReducedRanks) {
            int subCommRank, subCommNRank;
            NCCLCHECK(ncclCommUserRank(subComms[i], &subCommRank));
            NCCLCHECK(ncclCommCount(subComms[i], &subCommNRank));
        
            ASSERT_EQ(originalRank, subCommRank);
            ASSERT_EQ(subCommNRank, numReducedRanks);
        } else {
            ASSERT_EQ(subComms[i], nullptr);
        }
    }

    // Destroy sub-comms
    for (auto& subComm : subComms) {
        NCCLCHECK(ncclCommDestroy(subComm));
    }

    // Destroy original comms
    for (auto& comm : comms) {
        NCCLCHECK(ncclCommDestroy(comm));
    }
}