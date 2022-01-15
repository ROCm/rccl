/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <unistd.h>
#include "TestBed.hpp"
#include "TestChecks.hpp"
#include <rccl.h>

#define PIPE_WRITE(childId, val) \
  ASSERT_EQ(write(childList[childId]->parentWriteFd, &val, sizeof(val)), sizeof(val))

#define PIPE_CHECK(childId)                                             \
  {                                                                     \
    int response = 0;                                                   \
    ASSERT_EQ(read(childList[childId]->parentReadFd, &response, sizeof(int)), sizeof(int)); \
    ASSERT_EQ(response, TEST_SUCCESS);                                  \
  }

namespace RcclUnitTesting
{
  TestBed::TestBed() :
    numDevicesAvailable(0),
    numActiveChildren(0),
    numActiveRanks(0)
  {
    // Check for verbose env var
    this->verbose = getenv("UT_VERBOSE");

    // Set NCCL_COMM_ID to use a local port to avoid passing ncclCommId
    // Calling ncclGetUniqueId would initialize HIP, which should not be done prior to fork
    std::string localPort = "55513";
    if (!getenv("NCCL_COMM_ID"))
    {
      char hostname[HOST_NAME_MAX+1];
      gethostname(hostname, HOST_NAME_MAX+1);
      std::string hostnameString(hostname);
      hostnameString.append(":55513");
      setenv("NCCL_COMM_ID", hostnameString.c_str(), 0);
      if (this->verbose) printf("NCCL_COMM_ID set to %s\n", hostnameString.c_str());
    }

    // Collect the number of GPUs
    // NOTE: Cannot use HIP call prior to launching child processes via fork so use HSA
    this->numDevicesAvailable = 0;
    hsa_init();
    hsa_iterate_agents(TestBed::CountGpus, &this->numDevicesAvailable);
    hsa_shut_down();
    if (this->verbose) printf("Detected %d GPUs\n", this->numDevicesAvailable);

    // Create the maximum number of possible child processes (1 per GPU)
    // Parent and child communicate via pipes
    childList.resize(this->numDevicesAvailable);
    for (int childId = 0; childId < this->numDevicesAvailable; ++childId)
    {
      childList[childId] = new TestBedChild(childId, this->verbose);
      if (childList[childId]->InitPipes() != TEST_SUCCESS)
      {
        printf("[ERROR] Unable to create pipes to child process\n");
        return;
      }

      pid_t pid = fork();
      if (pid == 0)
      {
        // Child process enters execution loop
        childList[childId]->StartExecutionLoop();
        return;
      }
      else
      {
        // Parent records child process ID and closes unused ends of pipe
        childList[childId]->pid = pid;
        close(childList[childId]->childWriteFd);
        close(childList[childId]->childReadFd);
      }
    }
  }

  void TestBed::InitComms(std::vector<std::vector<int>> const& deviceIdsPerProcess,
                          int const numCollectivesInGroup)
  {
    // Count up the total number of GPUs to use and track child/deviceId per rank
    this->numActiveChildren = deviceIdsPerProcess.size();
    this->numActiveRanks = 0;
    this->numCollectivesInGroup = numCollectivesInGroup;
    this->rankToChildMap.clear();
    this->rankToDeviceMap.clear();
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      for (auto i = 0; i < deviceIdsPerProcess[childId].size(); ++i)
      {
        this->rankToChildMap.push_back(childId);
        this->rankToDeviceMap.push_back(deviceIdsPerProcess[childId][i]);
        ++this->numActiveRanks;
      }
    }

    // Send InitComms command to each active child process
    int const cmd = TestBedChild::CHILD_INIT_COMMS;
    int rankOffset = 0;
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      PIPE_WRITE(childId, cmd);

      // Send total number of ranks to child process
      PIPE_WRITE(childId, this->numActiveRanks);

      // Send the rank offset for this child process
      PIPE_WRITE(childId, rankOffset);

      // Send the number of collectives to be run per group call
      PIPE_WRITE(childId, numCollectivesInGroup);

      // Send the GPUs this child uses
      int const numGpus = deviceIdsPerProcess[childId].size();
      PIPE_WRITE(childId, numGpus);
      for (int i = 0; i < numGpus; i++)
        PIPE_WRITE(childId, deviceIdsPerProcess[childId][i]);

      rankOffset += numGpus;
    }

    // Wait for child acknowledgement
    // This is done after previous loop to avoid deadlock as every rank needs to enter ncclInitCommRank
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      PIPE_CHECK(childId);
    }
  }

  void TestBed::InitComms(int const numGpus, int const numCollectivesInGroup)
  {
    InitComms(TestBed::GetDeviceIdsList(1, numGpus), numCollectivesInGroup);
  }

  void TestBed::SetCollectiveArgs(ncclFunc_t     const funcType,
                                  ncclDataType_t const dataType,
                                  ncclRedOp_t    const redOp,
                                  int            const root,
                                  size_t         const numInputElements,
                                  size_t         const numOutputElements,
                                  int            const collId,
                                  int            const rank)
  {
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    // Loop over all ranks and send CollectiveArgs to appropriate child process
    int const cmd = TestBedChild::CHILD_SET_COLL_ARGS;
    for (auto currRank : rankList)
    {
      int const childId = rankToChildMap[currRank];
      PIPE_WRITE(childId, cmd);
      PIPE_WRITE(childId, currRank);
      PIPE_WRITE(childId, collId);
      PIPE_WRITE(childId, funcType);
      PIPE_WRITE(childId, dataType);
      PIPE_WRITE(childId, redOp);
      PIPE_WRITE(childId, root);
      PIPE_WRITE(childId, numInputElements);
      PIPE_WRITE(childId, numOutputElements);
      PIPE_CHECK(childId);
    }
  }

  void TestBed::AllocateMem(size_t const numInputBytes,
                            size_t const numOutputBytes,
                            bool   const inPlace,
                            bool   const useManagedMem,
                            int    const collId,
                            int    const rank)
  {
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    // Loop over all ranks and send allocation command to appropriate child process
    int const cmd = TestBedChild::CHILD_ALLOCATE_MEM;
    for (auto currRank : rankList)
    {
      int const childId = rankToChildMap[currRank];
      PIPE_WRITE(childId, cmd);
      PIPE_WRITE(childId, currRank);
      PIPE_WRITE(childId, collId);
      PIPE_WRITE(childId, numInputBytes);
      PIPE_WRITE(childId, numOutputBytes);
      PIPE_WRITE(childId, inPlace);
      PIPE_WRITE(childId, useManagedMem);
      PIPE_CHECK(childId);
    }
  }

  void TestBed::PrepareData(int         const collId,
                            int         const rank,
                            CollFuncPtr const prepDataFunc)
  {
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    // Loop over all ranks and send prepare data command to appropriate child process
    int const cmd = TestBedChild::CHILD_PREPARE_DATA;
    for (auto currRank : rankList)
    {
      int const childId = rankToChildMap[currRank];
      PIPE_WRITE(childId, cmd);
      PIPE_WRITE(childId, currRank);
      PIPE_WRITE(childId, collId);
      PIPE_WRITE(childId, prepDataFunc);
      PIPE_CHECK(childId);
    }
  }

  void TestBed::ExecuteCollectives()
  {
    int const cmd = TestBedChild::CHILD_EXECUTE_COLL;
    // Send ExecuteColl command to each active child process
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      PIPE_WRITE(childId, cmd);
    }

    // Wait for child acknowledgement
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      PIPE_CHECK(childId);
    }
  }

  void TestBed::ValidateResults(int collId, int const rank)
  {
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    int const cmd = TestBedChild::CHILD_VALIDATE_RESULTS;

    // Send ValidateResults command to each active child process
    for (auto currRank : rankList)
    {
      int const childId = rankToChildMap[currRank];
      PIPE_WRITE(childId, cmd);
      PIPE_WRITE(childId, currRank);
      PIPE_WRITE(childId, collId);
      PIPE_CHECK(childId);
    }
  }

  void TestBed::DeallocateMem(int const collId, int const rank)
  {
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    int const cmd = TestBedChild::CHILD_DEALLOCATE_MEM;

    for (auto currRank : rankList)
    {
      int const childId = rankToChildMap[currRank];
      PIPE_WRITE(childId, cmd);
      PIPE_WRITE(childId, currRank);
      PIPE_WRITE(childId, collId);
      PIPE_CHECK(childId);
    }
  }

  void TestBed::DestroyComms()
  {
    int const cmd = TestBedChild::CHILD_DESTROY_COMMS;
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      // Send DestroyComms command to each active child process
      PIPE_WRITE(childId, cmd);

      // Wait for child acknowledgement
      PIPE_CHECK(childId);
    }

    // Reset bookkeeping
    this->numActiveChildren = 0;
    this->numActiveRanks = 0;
    this->numCollectivesInGroup = 0;
  }

  void TestBed::Finalize()
  {
    // Send Stop to all child processes
    int const cmd = TestBedChild::CHILD_STOP;
    for (int childId = 0; childId < this->numDevicesAvailable; ++childId)
    {
      PIPE_WRITE(childId, cmd);

      // Close pipes to child process
      close(childList[childId]->parentWriteFd);
      close(childList[childId]->parentReadFd);
    }
    this->numDevicesAvailable = 0;
  }

  TestBed::~TestBed()
  {
    Finalize();
  }

  hsa_status_t TestBed::CountGpus(hsa_agent_t agent, void* data)
  {
    int* currCount = (int*)data;
    hsa_device_type_t device;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device);
    if (device == HSA_DEVICE_TYPE_GPU)
      *currCount = *currCount + 1;
    return HSA_STATUS_SUCCESS;
  }


  std::vector<ncclRedOp_t> TestBed::GetAllSupportedRedOps()
  {
    std::vector<ncclRedOp_t> ops;
    ops.push_back(ncclSum);

    // Skip non-sum reduction operators if only AllReduce is being built
#ifndef BUILD_ALLREDUCE_ONLY
    ops.push_back(ncclProd);
    ops.push_back(ncclMax);
    ops.push_back(ncclMin);
    ops.push_back(ncclAvg);
#endif
    return ops;
  }

  std::vector<ncclDataType_t> TestBed::GetAllSupportedDataTypes()
  {
    std::vector<ncclDataType_t> datatypes;
    datatypes.push_back(ncclFloat32);

    // Skip all but 32-bit floats if only AllReduce is being built
#ifndef BUILD_ALLREDUCE_ONLY
    datatypes.push_back(ncclInt8);
    datatypes.push_back(ncclUint8);
    datatypes.push_back(ncclInt32);
    datatypes.push_back(ncclUint32);
    datatypes.push_back(ncclInt64);
    datatypes.push_back(ncclUint64);
    // Half-precision floats disabled due to lack of host-side support
    // datatypes.push_back(ncclFloat16);
    datatypes.push_back(ncclFloat32);
    datatypes.push_back(ncclFloat64);
    datatypes.push_back(ncclBfloat16);
#endif
    return datatypes;
  }


  std::vector<std::vector<int>> TestBed::GetDeviceIdsList(int const numProcesses,
                                                 int const numGpus)
  {
    std::vector<std::vector<int>> result(numProcesses);
    for (int i = 0; i < numGpus; i++)
      result[i % numProcesses].push_back(i);
    return result;
  }
}

#undef PIPE_WRITE
