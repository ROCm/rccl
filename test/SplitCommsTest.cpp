/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include "StandaloneUtils.hpp"

class SplitCommsTest : public ::testing::Test {
protected:
    std::vector<ncclComm_t> comms;
    int numDevices = 3;

    void SetUp() override {
        //HIPCALL(hipGetDeviceCount(&numDevices));

        if (numDevices < 2) {
            GTEST_SKIP() << "This test requires at least 2 GPUs.";
        }

        ncclUniqueId id;
        comms.resize(numDevices);

        NCCLCHECK(ncclGetUniqueId(&id));

        // Initialize the original comms
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < numDevices; i++) {
            HIPCALL(hipSetDevice(i));
            NCCLCHECK(ncclCommInitRank(&comms[i], numDevices, id, i));
        }
        NCCLCHECK(ncclGroupEnd());
    }

    void TearDown() override {
        // Destroy the original comms after the test ends
        for (auto& comm : comms) {
            NCCLCHECK(ncclCommDestroy(comm));
        }
    }
};

TEST_F(SplitCommsTest, RankCheck) {
    std::vector<ncclComm_t> subComms(numDevices);
    int numSubComms = 2;

    // Split the original comms into sub comms
    NCCLCHECK(ncclGroupStart());
    for (int localRank = 0; localRank < numDevices; localRank++) {
        NCCLCHECK(ncclCommSplit(comms[localRank], localRank % numSubComms, localRank, &subComms[localRank], NULL));
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

        // Calculate the expected sub-communication properties
        int maxExpectedSubCommNRank = (comms.size() + numSubComms - 1) / numSubComms;
        int minExpectedSubCommNRank = maxExpectedSubCommNRank - ((i % numSubComms == 0) ? 0 : 1);

        // Assert sub-communication properties
        ASSERT_EQ(subCommRank, i / numSubComms);
        ASSERT_GE(subCommNRank, minExpectedSubCommNRank);
        ASSERT_LE(subCommNRank, maxExpectedSubCommNRank);
    }

    // Destroy the created sub comms
    for (auto& subComm : subComms) {
        NCCLCHECK(ncclCommDestroy(subComm));
    }
}

// Test to duplicate the original comm
TEST_F(SplitCommsTest, OneColor) {
    std::vector<ncclComm_t> subComms(numDevices);

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
        
        ASSERT_EQ(originalNRank, subCommNRank);
        ASSERT_EQ(originalNRank, subCommNRank);
    }

    for (auto& subComm : subComms) {
        NCCLCHECK(ncclCommDestroy(subComm));
    }
}

TEST_F(SplitCommsTest, ReduceRanks) {
    std::vector<ncclComm_t> subComms(numDevices);

    // Number of ranks to duplicate from the original comm
    int numReducedRanks = numDevices / 2;

    NCCLCHECK(ncclGroupStart());
    for (int localRank = 0; localRank < numDevices; localRank++) {
        NCCLCHECK(ncclCommSplit(comms[localRank], localRank < numReducedRanks ? 0 : NCCL_SPLIT_NOCOLOR, localRank, &subComms[localRank], NULL));
    }
    NCCLCHECK(ncclGroupEnd());

    // Validate results
    for (int i = 0; i < numReducedRanks; i++) {
        int originalRank, originalNRank;
        NCCLCHECK(ncclCommUserRank(comms[i], &originalRank));
        NCCLCHECK(ncclCommCount(comms[i], &originalNRank));

        int subCommRank, subCommNRank;
        NCCLCHECK(ncclCommUserRank(subComms[i], &subCommRank));
        NCCLCHECK(ncclCommCount(subComms[i], &subCommNRank));
        
        ASSERT_EQ(originalRank, subCommRank);
        ASSERT_EQ(subCommNRank, numReducedRanks);
    }

    // Destroy subComms
    for (auto& subComm : subComms) {
        NCCLCHECK(ncclCommDestroy(subComm));
    }
}