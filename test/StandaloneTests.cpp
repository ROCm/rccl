/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include "StandaloneUtils.hpp"

namespace RcclUnitTesting {
  TEST(Standalone, SplitComms_RankCheck)
  {
    // Check for multi-gpu
    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    // Initialize the original comms
    std::vector<ncclComm_t> comms(numDevices);
    NCCLCHECK(ncclCommInitAll(comms.data(), numDevices, nullptr));

    // Split into new comms (round-robin)
    std::vector<ncclComm_t> subComms(numDevices);
    int numSubComms = 2;

    std::map<int, int> mapCounter;
    NCCLCHECK(ncclGroupStart());
    for (int localRank = 0; localRank < numDevices; localRank++) {
      NCCLCHECK(ncclCommSplit(comms[localRank], localRank % numSubComms, localRank, &subComms[localRank], NULL));
      mapCounter[localRank % numSubComms]++;
    }
    NCCLCHECK(ncclGroupEnd());

    // Check that new comms have correct subranks / ranks
    for (int i = 0; i < numDevices; i++) {
      int subCommRank, subCommNRank;
      NCCLCHECK(ncclCommUserRank(subComms[i], &subCommRank));
      NCCLCHECK(ncclCommCount(subComms[i], &subCommNRank));

      ASSERT_EQ(subCommRank, i / numSubComms);
      ASSERT_EQ(subCommNRank, mapCounter[i % numSubComms]);
    }

    // Clean up comms
    for (auto& subComm : subComms)
      NCCLCHECK(ncclCommDestroy(subComm));
    for (auto& comm : comms)
      NCCLCHECK(ncclCommDestroy(comm));
  }

  TEST(Standalone, SplitComms_OneColor)
  {
    // Check for multi-gpu
    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    // Initialize the original comms
    std::vector<ncclComm_t> comms(numDevices);
    NCCLCHECK(ncclCommInitAll(comms.data(), numDevices, nullptr));

    // Split into new comms (all of the same color)
    std::vector<ncclComm_t> subComms(numDevices);  
    NCCLCHECK(ncclGroupStart());
    for (int localRank = 0; localRank < numDevices; localRank++)
      NCCLCHECK(ncclCommSplit(comms[localRank], 0, localRank, &subComms[localRank], NULL));
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

    // Clean up comms
    for (auto& subComm : subComms)
      NCCLCHECK(ncclCommDestroy(subComm));
    for (auto& comm : comms)
      NCCLCHECK(ncclCommDestroy(comm));
  }

  TEST(Standalone, SplitComms_Reduce)
  {
    // Check for multi-gpu
    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    // Initialize the original comms
    std::vector<ncclComm_t> comms(numDevices);
    NCCLCHECK(ncclCommInitAll(comms.data(), numDevices, nullptr));

    // Split into new comms
    int numReducedRanks = numDevices / 2; 
    std::vector<ncclComm_t> subComms(numDevices);
    NCCLCHECK(ncclGroupStart());
    for (int localRank = 0; localRank < numDevices; localRank++)
      NCCLCHECK(ncclCommSplit(comms[localRank],
            localRank < numReducedRanks ? 0 : NCCL_SPLIT_NOCOLOR,
            localRank, &subComms[localRank], NULL));
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

    // Cleanup comms
    for (auto& subComm : subComms)
      NCCLCHECK(ncclCommDestroy(subComm));
    for (auto& comm : comms)
      NCCLCHECK(ncclCommDestroy(comm));
  }
}