/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include "TestBed.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{
  /**
   * \brief Verify that each device is assigned to the right rank using ncclCommSplit API.
   * ******************************************************************************************/
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

  /**
   * \brief Creates a communicator for each device and gathers them all in one rank.
   * ******************************************************************************************/
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

  /**
   * \brief Creates a communicator for each device and reduces them into (numDevices / 2) ranks.
   * ******************************************************************************************/
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

  /**
   * \brief Verify there is no regression in timing for each protocol [LL, LL128, Simple]
   * ******************************************************************************************/
  TEST(Standalone, RegressionTiming)
  {
    TestBed testBed;
    // timing
    using namespace std::chrono;
    using Clock = std::chrono::high_resolution_clock;
    int usElapsed, numIterations = 20, numWarmups = 5;

    // Check for 2 GPUs
    int numGpus;
    HIPCALL(hipGetDeviceCount(&numGpus));
    if (numGpus < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    // Initialize RCCL
    int numRanks = 2;
    std::vector<ncclComm_t> comms(numRanks);

    char *proto = std::getenv("NCCL_PROTO");
    const char* protocolList[3] = {"LL", "LL128", "Simple"};

    for (auto p : protocolList)
    {
      usElapsed = 0;
      if(testBed.ev.isGfx12) {
        setenv("NCCL_PROTO", "Simple", 1);
      } else {
        setenv("NCCL_PROTO", p, 1);
      }

      NCCLCHECK(ncclCommInitAll(comms.data(), numRanks, nullptr));

      // Prepare CPU data arrays
      int N = 1250;
      std::vector<int> cpuInput(N);
      std::vector<int> cpuExpected(N);
      for (int i = 0; i < N; i++) {
        cpuInput[i]    = i;
        cpuExpected[i] = 2 * i;
      }

      // Prepare GPU data arrays
      int* gpuInput[numRanks];
      int* gpuOutput[numRanks];
      hipStream_t stream[numRanks];

      for (int rank = 0; rank < numRanks; rank++) {
        HIPCALL(hipSetDevice(rank));
        HIPCALL(hipStreamCreate(&stream[rank]));
        HIPCALL(hipMalloc((void**)&gpuInput[rank], N * sizeof(int)));
        HIPCALL(hipMalloc((void**)&gpuOutput[rank], N * sizeof(int)));
        HIPCALL(hipMemcpy(gpuInput[rank], cpuInput.data(), N * sizeof(int), hipMemcpyHostToDevice));
        HIPCALL(hipMemset(gpuOutput[rank], 0, N * sizeof(int)));
        HIPCALL(hipDeviceSynchronize());
      }

      for (int iter = -numWarmups; iter < numIterations; iter++) {

        for (int rank = 0; rank < numRanks; rank++) {
          HIPCALL(hipSetDevice(rank));
          HIPCALL(hipMemset(gpuOutput[rank], 0, N * sizeof(int)));
          HIPCALL(hipDeviceSynchronize());
	}

        // Initiate the allreduce
        NCCLCHECK(ncclGroupStart());
        for (int rank = 0; rank < numRanks; rank++)
          NCCLCHECK(ncclAllReduce(gpuInput[rank], gpuOutput[rank], N, ncclInt, ncclSum, comms[rank], stream[rank]));
        NCCLCHECK(ncclGroupEnd());

        const auto start = Clock::now();

        // Wait for completion
        for (int rank = 0; rank < numRanks; rank++) {
          HIPCALL(hipStreamSynchronize(stream[rank]));
        }

	if (iter >= 0)
          usElapsed += duration_cast<microseconds>(Clock::now() - start).count();

        // Check results
        std::vector<int> cpuOutput(N);
        for (int rank = 0; rank < numRanks; rank++) {
          HIPCALL(hipMemcpy(cpuOutput.data(), gpuOutput[rank], N * sizeof(int), hipMemcpyDeviceToHost));
          HIPCALL(hipDeviceSynchronize());
          for (int i = 0; i < N; i++)
            ASSERT_EQ(cpuOutput[i], cpuExpected[i]);
        }
      }

      EXPECT_LT(usElapsed/(double)numIterations, 5000);
      printf("[ INFO     ] protocol: %s, average runtime: %f microseconds\n", p, usElapsed/(double)numIterations);
      // Release resources
      for (int rank = 0; rank < numRanks; rank++){
        HIPCALL(hipFree(gpuInput[rank]));
        HIPCALL(hipFree(gpuOutput[rank]));
        HIPCALL(hipStreamDestroy(stream[rank]));
        NCCLCHECK(ncclCommDestroy(comms[rank]));
      }
    }
    if (proto)
      setenv("NCCL_PROTO", proto, 1);
    else
      unsetenv("NCCL_PROTO");
  }

  /**
   * \brief Verify rccl generic kernel stack size for each gfx architecture is less than the
   * expected MAX_STACK_SIZE.
   * ******************************************************************************************/
  TEST(Standalone, StackSize) {
    const char* mainKernel = "ncclDevKernel";

    // Look for the .co files
    std::vector<std::string> coFileList = splitString(executeCommand("find ../ -type f -name \"*.co\""), '\n');

    // Check if the .co files exist in the build directory
    if (coFileList.empty())
      GTEST_SKIP() << "Skipping... Could not found required files in the build directory.";

    for (const auto& file : coFileList) {
      // Store the output in a list
      std::string cmd = std::string(ROCM_PATH) + "/llvm/bin/llvm-readelf --notes " + file;
      std::vector<std::string> metadata = splitString(executeCommand(cmd.c_str()), '\n');

      // Skip if llvm is not installed
      if (metadata.empty())
        GTEST_SKIP() << "Skipping... llvm is not found.";

      // Parse metadata from file and store it for each arch
      ArchInfo archInfo = parseMetadata(metadata);

      // iterate over each archs kernels
      for (const auto& kernel : archInfo.kernels) {
        if (kernel.name.find(mainKernel) != std::string::npos) {
          // Kernel stack size should be less than or equal to the maxStackSize value
          printf("[ INFO     ] Arch: %s Kernel: %s Size: %d\n", archInfo.archName.c_str(), kernel.name.c_str(), kernel.privateSegmentFixedSize);
          EXPECT_LE(kernel.privateSegmentFixedSize, archInfo.archName == "gfx90a" ? MAX_STACK_SIZE_gfx90a : MAX_STACK_SIZE);
        }
      }
    }
  }
}
