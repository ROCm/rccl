/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "TestBedChild.hpp"

#define CHILD_NCCL_CALL(cmd, msg) \
  {                                                                     \
    ncclResult_t status = cmd;                                          \
    if (status != ncclSuccess)                                          \
    {                                                                   \
      printf("[ERROR] Child %d fails %s code %d\n", this->childId, msg, status); \
      return TEST_FAIL;                                                 \
    }                                                                   \
  }

#define PIPE_READ(val) \
  if (read(childReadFd, &val, sizeof(val)) != sizeof(val)) return TEST_FAIL;

namespace RcclUnitTesting
{
  TestBedChild::TestBedChild(int const childId, bool const verbose)
  {
    this->childId = childId;
    this->verbose = verbose;
  }

  int TestBedChild::InitPipes()
  {
    // Prepare parent->child pipe
    int pipefd[2];
    if (pipe(pipefd) == -1)
    {
      printf("[ERROR] Unable to create parent->child pipe for child %d\n", this->childId);
      return TEST_FAIL;
    }
    this->childReadFd   = pipefd[0];
    this->parentWriteFd = pipefd[1];

    // Prepare child->parent pipe
    this->parentReadFd = -1;
    if (pipe(pipefd) == -1)
    {
      printf("[ERROR] Unable to create child->parent pipe for child %d\n", this->childId);
      return TEST_FAIL;
    }
    this->parentReadFd = pipefd[0];
    this->childWriteFd = pipefd[1];
    return TEST_SUCCESS;
  }

  void TestBedChild::StartExecutionLoop()
  {
    // Close unused ends of pipes
    close(this->parentWriteFd);
    close(this->parentReadFd);

    // Wait for commands from parent process
    if (verbose) printf("Child %d enters execution loop\n", this->childId);
    int command;
    while (read(childReadFd, &command, sizeof(command)) > 0)
    {
      if (verbose) printf("Child %d received command %d:\n", childId, command);
      ErrCode status = TEST_SUCCESS;
      switch(command)
      {
      case CHILD_INIT_COMMS      : status = InitComms();          break;
      case CHILD_SET_COLL_ARGS   : status = SetCollectiveArgs();  break;
      case CHILD_ALLOCATE_MEM    : status = AllocateMem();        break;
      case CHILD_PREPARE_DATA    : status = PrepareData();        break;
      case CHILD_EXECUTE_COLL    : status = ExecuteCollectives(); break;
      case CHILD_VALIDATE_RESULTS: status = ValidateResults();    break;
      case CHILD_DEALLOCATE_MEM  : status = DeallocateMem();      break;
      case CHILD_DESTROY_COMMS   : status = DestroyComms();       break;
      case CHILD_STOP            : status = Stop();               break;
      default: exit(0);
      }

      // Send back acknowledgement to parent
      write(childWriteFd, &status, sizeof(status));
    }

    // Close child ends of pipe
    close(this->childReadFd);
    close(this->childWriteFd);

    exit(0);
  }

  ErrCode TestBedChild::InitComms()
  {
    if (this->verbose) printf("Child %d begins InitComms()\n", this->childId);

    // Read values sent by parent [see TestBed::InitComms()]
    PIPE_READ(this->totalRanks);
    PIPE_READ(this->rankOffset);
    PIPE_READ(this->numCollectivesInGroup);

    // Read the GPUs this child uses and prepare storage for collective args / datasets
    int numGpus;
    PIPE_READ(numGpus);
    this->deviceIds.resize(numGpus);
    this->streams.resize(numGpus);
    this->collArgs.resize(numGpus);
    for (int i = 0; i < numGpus; i++)
    {
      PIPE_READ(this->deviceIds[i]);
      this->collArgs[i].clear();
      this->collArgs[i].resize(numCollectivesInGroup);
    }

    // Collect uniqueId (specified by NCCL_COMM_ID env var)
    ncclUniqueId id;
    CHILD_NCCL_CALL(ncclGetUniqueId(&id), "ncclGetUniqueId");

    // Initialize communicators
    comms.clear();
    comms.resize(numGpus);
    ErrCode status = TEST_SUCCESS;

    // NOTE: Using multiple threads in order to initialize communicators
    //       Using ncclGroupStart / ncclGroupEnd with NCCL_COMM_ID results in errors
    #pragma omp parallel for num_threads(numGpus)
    for (int i = 0; i < numGpus; i++)
    {
      int const currGpu = this->deviceIds[i];
      if (status == TEST_SUCCESS && hipSetDevice(currGpu) != hipSuccess)
      {
        printf("[ERROR] Child %d unable to set device to %d\n", this->childId, currGpu);
        status = TEST_FAIL;
      }

      if (status == TEST_SUCCESS && hipStreamCreate(&streams[i]) != hipSuccess)
      {
        printf("[ERROR] Child %d unable to create stream for GPU %d\n", this->childId, currGpu);
        status = TEST_FAIL;
      }

      if (status == TEST_SUCCESS && ncclCommInitRank(&comms[i], totalRanks, id, rankOffset + i) != ncclSuccess)
      {
        printf("[ERROR] Child %d unable to call ncclCommInitRank for rank %d\n", this->childId, rankOffset + i);
        status = TEST_FAIL;
      }
    }
    if (status != TEST_SUCCESS) return status;

    if (this->verbose) printf("Child %d finishes InitComms()\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::SetCollectiveArgs()
  {
    if (this->verbose) printf("Child %d begins SetCollectiveArgs\n", this->childId);

    // Read values sent by parent [see TestBed::SetCollectiveArgs()]
    int            globalRank;
    int            collId;
    ncclFunc_t     funcType;
    ncclDataType_t dataType;
    ncclRedOp_t    redOp;
    int            root;
    size_t         numInputElements;
    size_t         numOutputElements;

    PIPE_READ(globalRank);
    PIPE_READ(collId);
    PIPE_READ(funcType);
    PIPE_READ(dataType);
    PIPE_READ(redOp);
    PIPE_READ(root);
    PIPE_READ(numInputElements);
    PIPE_READ(numOutputElements);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      printf("[ERROR] Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;

    for (int collIdx = 0; collIdx < collArgs.size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        CHECK_CALL(this->collArgs[localRank][collIdx].SetArgs(globalRank, this->totalRanks,
                                                              this->deviceIds[localRank],
                                                              funcType, dataType, redOp, root,
                                                              numInputElements, numOutputElements));
      }
    }

    if (this->verbose) printf("Child %d finishes SetCollectiveArgs\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::AllocateMem()
  {
    if (this->verbose) printf("Child %d begins AllocateMem\n", this->childId);

    // Read values sent by parent [see TestBed::AllocateMem()]
    int    globalRank;
    int    collId;
    size_t numInputBytes;
    size_t numOutputBytes;
    bool   inPlace;
    bool   useManagedMem;

    PIPE_READ(globalRank);
    PIPE_READ(collId);
    PIPE_READ(numInputBytes);
    PIPE_READ(numOutputBytes);
    PIPE_READ(inPlace);
    PIPE_READ(useManagedMem);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      printf("[ERROR] Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;

    for (int collIdx = 0; collIdx < collArgs.size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        CHECK_CALL(this->collArgs[localRank][collIdx].AllocateMem(numInputBytes, numOutputBytes, inPlace, useManagedMem));
      }
    }

    if (this->verbose) printf("Child %d finishes AllocateMem\n", this->childId);
    return TEST_SUCCESS;
  }

  // Fill input memory with pre-known patterned based on rank
  ErrCode TestBedChild::PrepareData()
  {
    if (this->verbose) printf("Child %d begins PrepareData\n", this->childId);

    // Read values sent by parent [see TestBed::PrepareData()]
    int globalRank;
    int collId;
    CollFuncPtr prepDataFunc;

    PIPE_READ(globalRank);
    PIPE_READ(collId);
    PIPE_READ(prepDataFunc);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      printf("[ERROR] Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;

    for (int collIdx = 0; collIdx < collArgs.size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        CHECK_CALL(this->collArgs[localRank][collIdx].PrepareData(prepDataFunc));
      }
    }
    if (this->verbose) printf("Child %d finishes FillPattern\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::ExecuteCollectives()
  {
    if (this->verbose) printf("Child %d begins ExecuteCollectives\n", this->childId);

    // Start group call
    CHILD_NCCL_CALL(ncclGroupStart(), "ncclGroupStart");

    // Loop over all collectives to be executed in group call
    for (int collId = 0; collId < this->numCollectivesInGroup; ++collId)
    {
      // Loop over all local ranks
      for (int localRank = 0; localRank < this->deviceIds.size(); ++localRank)
      {
        CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

        CollectiveArgs const& collArg = this->collArgs[localRank][collId];
        switch (collArg.funcType)
        {
        case ncclCollBroadcast:
          CHILD_NCCL_CALL(ncclBroadcast(collArg.inputGpu.ptr,
                                        collArg.outputGpu.ptr,
                                        collArg.numInputElements,
                                        collArg.dataType,
                                        collArg.root,
                                        this->comms[localRank],
                                        this->streams[localRank]),
                          "ncclBroadcast");
          break;
        case ncclCollReduce:
          CHILD_NCCL_CALL(ncclReduce(collArg.inputGpu.ptr,
                                     collArg.outputGpu.ptr,
                                     collArg.numInputElements,
                                     collArg.dataType,
                                     collArg.redOp,
                                     collArg.root,
                                     this->comms[localRank],
                                     this->streams[localRank]),
                          "ncclReduce");
          break;
        case ncclCollAllGather:
          CHILD_NCCL_CALL(ncclAllGather(collArg.inputGpu.ptr,
                                        collArg.outputGpu.ptr,
                                        collArg.numInputElements,
                                        collArg.dataType,
                                        this->comms[localRank],
                                        this->streams[localRank]),
                          "ncclAllGather");
          break;
        case ncclCollReduceScatter:
          CHILD_NCCL_CALL(ncclReduceScatter(collArg.inputGpu.ptr,
                                            collArg.outputGpu.ptr,
                                            collArg.numOutputElements,
                                            collArg.dataType,
                                            collArg.redOp,
                                            this->comms[localRank],
                                            this->streams[localRank]),
                          "ncclReduceScatter");
          break;
        case ncclCollAllReduce:
          CHILD_NCCL_CALL(ncclAllReduce(collArg.inputGpu.ptr,
                                        collArg.outputGpu.ptr,
                                        collArg.numInputElements,
                                        collArg.dataType,
                                        collArg.redOp,
                                        this->comms[localRank],
                                        this->streams[localRank]),
                          "ncclAllReduce");
          break;
        case ncclCollGather:
          CHILD_NCCL_CALL(ncclGather(collArg.inputGpu.ptr,
                                     collArg.outputGpu.ptr,
                                     collArg.numInputElements,
                                     collArg.dataType,
                                     collArg.redOp,
                                     this->comms[localRank],
                                     this->streams[localRank]),
                          "ncclGather");
          break;
        case ncclCollScatter:
          CHILD_NCCL_CALL(ncclScatter(collArg.inputGpu.ptr,
                                      collArg.outputGpu.ptr,
                                      collArg.numOutputElements,
                                      collArg.dataType,
                                      collArg.root,
                                      this->comms[localRank],
                                      this->streams[localRank]),
                          "ncclScatter");
          break;
        case ncclCollAllToAll:
          CHILD_NCCL_CALL(ncclAllToAll(collArg.inputGpu.ptr,
                                       collArg.outputGpu.ptr,
                                       collArg.numInputElements / collArg.totalRanks,
                                       collArg.dataType,
                                       this->comms[localRank],
                                       this->streams[localRank]),
                          "ncclAllToAll");
          break;
        case ncclCollSend:
          CHILD_NCCL_CALL(ncclSend(collArg.inputGpu.ptr,
                                   collArg.numInputElements,
                                   collArg.dataType,
                                   collArg.root,
                                   this->comms[localRank],
                                   this->streams[localRank]),
                          "ncclSend");
          break;
        case ncclCollRecv:
          CHILD_NCCL_CALL(ncclRecv(collArg.outputGpu.ptr,
                                   collArg.numOutputElements,
                                   collArg.dataType,
                                   collArg.root,
                                   this->comms[localRank],
                                   this->streams[localRank]),
                          "ncclRecv");
          break;
        default:
          printf("[ERROR] Unknown func type %d\n", collArg.funcType);
          return TEST_FAIL;
        }
      }
    }

    // End group call
    CHILD_NCCL_CALL(ncclGroupEnd(), "ncclGroupEnd");

    // Synchronize
    if (this->verbose) printf("Child %d submits group call.  Waiting for completion\n", this->childId);
    for (int localRank = 0; localRank < this->streams.size(); ++localRank)
    {
      CHECK_HIP(hipStreamSynchronize(this->streams[localRank]));
    }
    if (this->verbose) printf("Child %d finishes ExecuteCollectives\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::ValidateResults()
  {
    if (this->verbose) printf("Child %d begins ValidateResults\n", this->childId);

    // Read values sent by parent [see TestBed::ValidateResults()]
    int globalRank, collId;
    PIPE_READ(globalRank);
    PIPE_READ(collId);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      printf("[ERROR] Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;

    ErrCode status = TEST_SUCCESS;
    for (int collIdx = 0; collIdx < collArgs.size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        if (this->collArgs[localRank][collIdx].ValidateResults() != TEST_SUCCESS)
          status = TEST_FAIL;
      }
    }
    if (this->verbose) printf("Child %d finishes ValidateResults\n", this->childId);
    return status;
  }

  ErrCode TestBedChild::DeallocateMem()
  {
    if (this->verbose) printf("Child %d begins DeallocateMem\n", this->childId);

    // Read values sent by parent [see TestBed::DeallocateMem()]
    int globalRank, collId;
    PIPE_READ(globalRank);
    PIPE_READ(collId);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      printf("[ERROR] Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;

    for (int collIdx = 0; collIdx < collArgs.size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        CHECK_CALL(this->collArgs[localRank][collIdx].DeallocateMem());
      }
    }

    if (this->verbose) printf("Child %d finishes DeallocateMem\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::DestroyComms()
  {
    if (this->verbose) printf("Child %d begins DestroyComms\n", this->childId);
    // Release comms
    for (int i = 0; i < this->comms.size(); ++i)
    {
      CHILD_NCCL_CALL(ncclCommDestroy(this->comms[i]), "ncclCommDestroy");
    }
    for (int i = 0; i < this->streams.size(); ++i)
    {
      CHECK_HIP(hipStreamDestroy(this->streams[i]));
    }
    this->comms.clear();
    if (this->verbose) printf("Child %d finishes DestroyComms\n", this->childId);
    return TEST_SUCCESS;
  }


  ErrCode TestBedChild::Stop()
  {
    return TEST_SUCCESS;
  }
}
