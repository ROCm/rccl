/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <vector>
#include <unistd.h>
#include "CollectiveArgs.hpp"
#include "rccl.h"

#define MAX_RANKS 32
namespace RcclUnitTesting
{
  class TestBedChild
  {
  public:
    // These are commands that can be given to the child process
    enum
    {
      CHILD_INIT_COMMS       = 0,  // InitComms()
      CHILD_SET_COLL_ARGS    = 1,  // SetCollectiveArgs()
      CHILD_ALLOCATE_MEM     = 2,  // AllocateMem()
      CHILD_PREPARE_DATA     = 3,  // PrepareData()
      CHILD_EXECUTE_COLL     = 4,  // ExecuteCollectives()
      CHILD_VALIDATE_RESULTS = 5,  // ValidateResults()
      CHILD_DEALLOCATE_MEM   = 6,  // DeallocateMem()
      CHILD_DESTROY_COMMS    = 7,  // DestroyComms()
      CHILD_STOP             = 8,  // Stop()
      NUM_CHILD_COMMANDS     = 9
    };

    char const ChildCommandNames[NUM_CHILD_COMMANDS][20] =
    {
      "INIT_COMMS",
      "SET_COLL_ARGS",
      "ALLOCATE_MEM",
      "PREPARE_DATA",
      "EXECUTE_COLL",
      "VALIDATE_RESULTS",
      "DEALLOCATE_MEM",
      "DESTROY_COMMS",
      "STOP"
    };

    // These variables remain constant for life of TestBedChild
    int   childId;
    pid_t pid;
    bool  verbose;
    int   printValues;

    // Pipes used to communicate between parent process
    int parentWriteFd;
    int parentReadFd;
    int childWriteFd;
    int childReadFd;

    // These varibles may change based on commands issued by parent
    int totalRanks;                                     // Total ranks
    int rankOffset;                                     // Global rank offset for this child
    int numCollectivesInGroup;                          // # of collectives to run per group call
    std::vector<ncclComm_t> comms;                      // RCCL communicators for each rank
    std::vector<int> deviceIds;                         // Device IDs for each rank
    std::vector<hipStream_t> streams;                   // Streams for executing collectives
    std::vector<std::vector<CollectiveArgs>> collArgs;  // Info for each collective for each rank

    // Constructor
    TestBedChild(int const childId, bool const verbose, int const printValues);

    // Prepare parent/child communication pipes - to be executed by parent process
    int InitPipes();

    // Execution
    void StartExecutionLoop();

  protected:
    // Initialize RCCL communicators
    ErrCode InitComms();

    // Set CollectiveArgs
    ErrCode SetCollectiveArgs();

    // Allocate memory (input (GPU) / output (GPU) / expected (CPU))
    ErrCode AllocateMem();

    // Prepare input and expected data
    ErrCode PrepareData();

    // Execute a group of collectives
    ErrCode ExecuteCollectives();

    // Validate that output matches expected
    ErrCode ValidateResults();

    // Release allocated memory
    ErrCode DeallocateMem();

    // Destroys RCCL communicators
    ErrCode DestroyComms();

    // Stops this child process
    ErrCode Stop();
  };
}
