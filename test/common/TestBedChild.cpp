/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "TestBedChild.hpp"

#include <thread>
#include <execinfo.h>

#define CHILD_NCCL_CALL(cmd, msg)                                       \
  {                                                                     \
    if (this->verbose) printf("[ NCCL CALL] " #cmd "\n");               \
    ncclResult_t status = cmd;                                          \
    if (status != ncclSuccess)                                          \
    {                                                                   \
      ERROR("Child process %d fails NCCL call %s with code %d\n", this->childId, msg, status); \
      return TEST_FAIL;                                                 \
    }                                                                   \
  }

#define PIPE_READ(val) \
  if (read(childReadFd, &val, sizeof(val)) != sizeof(val)) return TEST_FAIL;

namespace RcclUnitTesting
{
  TestBedChild::TestBedChild(int const childId, bool const verbose, int const printValues)
  {
    this->childId = childId;
    this->verbose = verbose;
    this->printValues = printValues;
  }

  int TestBedChild::InitPipes()
  {
    // Prepare parent->child pipe
    int pipefd[2];
    if (pipe(pipefd) == -1)
    {
      ERROR("Unable to create parent->child pipe for child %d\n", this->childId);
      return TEST_FAIL;
    }
    this->childReadFd   = pipefd[0];
    this->parentWriteFd = pipefd[1];

    // Prepare child->parent pipe
    this->parentReadFd = -1;
    if (pipe(pipefd) == -1)
    {
      ERROR("Unable to create parent->child pipe for child %d\n", this->childId);
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
    if (verbose) INFO("Child %d enters execution loop\n", this->childId);
    int command;
    while (read(childReadFd, &command, sizeof(command)) > 0)
    {
      if (verbose) INFO("Child %d received command [%s]:\n", this->childId, ChildCommandNames[command]);;
      ErrCode status = TEST_SUCCESS;
      switch(command)
      {
      case CHILD_GET_UNIQUE_ID   : status = GetUniqueId();        break;
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
      if (status == TEST_FAIL)
        ERROR("Child %d failed on command [%s]:\n", this->childId, ChildCommandNames[command]);
      if (write(childWriteFd, &status, sizeof(status)) < 0)
      {
        ERROR("Child %d write to parent failed: %s\n", this->childId, strerror(errno));
        break;
      }
    }
    if (verbose) INFO("Child %d exiting execution loop\n", this->childId);

    // Close child ends of pipe
    close(this->childReadFd);
    close(this->childWriteFd);

    exit(0);
  }

  ErrCode TestBedChild::GetUniqueId()
  {
    if (this->verbose) INFO("Child %d begins GetUniqueId()\n", this->childId);

    // Get a unique ID and pass it back to parent process
    ncclUniqueId id;
    CHILD_NCCL_CALL(ncclGetUniqueId(&id), "ncclGetUniqueId");
    write(childWriteFd, &id, sizeof(id));

    if (this->verbose) INFO("Child %d finishes GetUniqueId()\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::InitComms()
  {
    if (this->verbose) INFO("Child %d begins InitComms()\n", this->childId);

    // Read values sent by parent [see TestBed::InitComms()]
    ncclUniqueId id;
    PIPE_READ(id);
    PIPE_READ(this->totalRanks);
    PIPE_READ(this->rankOffset);
    PIPE_READ(this->numCollectivesInGroup);
    bool useMultiRankPerGpu;
    PIPE_READ(useMultiRankPerGpu);

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

    // Initialize communicators
    comms.clear();
    comms.resize(numGpus);

    // Initialize within a group call to avoid deadlock when using multiple ranks per child
    ErrCode status = TEST_SUCCESS;
    CHILD_NCCL_CALL(ncclGroupStart(), "ncclGroupStart");
    for (int localRank = 0; localRank < numGpus; ++localRank)
    {
      int const globalRank = this->rankOffset + localRank;
      int const currGpu = this->deviceIds[localRank];

      if (hipSetDevice(currGpu) != hipSuccess)
      {
        ERROR("Rank %d on child %d unable to switch to GPU %d\n", globalRank, this->childId, currGpu);
        status = TEST_FAIL;
        break;
      }

      if (hipStreamCreate(&this->streams[localRank]) != hipSuccess)
      {
        ERROR("Rank %d on child %d unable to create stream for GPU %d\n", globalRank, this->childId, currGpu);
        status = TEST_FAIL;
        break;
      }

      if (useMultiRankPerGpu)
      {
        if (ncclCommInitRankMulti(&this->comms[localRank], this->totalRanks, id, globalRank, globalRank) != ncclSuccess)
        {
          ERROR("Rank %d on child %d unable to call ncclCommInitRankMulti\n", globalRank, this->childId);
          status = TEST_FAIL;
          break;
        }
      }
      else
      {
        if (ncclCommInitRank(&this->comms[localRank], this->totalRanks, id, globalRank) != ncclSuccess)
        {
          ERROR("Rank %d on child %d unable to call ncclCommInitRank\n", globalRank, this->childId);
          status = TEST_FAIL;
          break;
        }
      }
    }
    if (status == TEST_SUCCESS)
    {
      CHILD_NCCL_CALL(ncclGroupEnd(), "ncclGroupStart");
    }
    if (this->verbose) INFO("Child %d finishes InitComms() [%s]\n",
                            this->childId, status == TEST_SUCCESS ? "SUCCESS" : "FAIL");
    return status;
  }

  ErrCode TestBedChild::SetCollectiveArgs()
  {
    if (this->verbose) INFO("Child %d begins SetCollectiveArgs()\n", this->childId);

    // Read values sent by parent [see TestBed::SetCollectiveArgs()]
    int             globalRank;
    int             collId;
    ncclFunc_t      funcType;
    ncclDataType_t  dataType;
    size_t          numInputElements;
    size_t          numOutputElements;
    OptionalColArgs options;

    PIPE_READ(globalRank);
    PIPE_READ(collId);
    PIPE_READ(funcType);
    PIPE_READ(dataType);
    PIPE_READ(numInputElements);
    PIPE_READ(numOutputElements);
    PIPE_READ(options);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      ERROR("Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

    for (int collIdx = 0; collIdx < collArgs[localRank].size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        CollectiveArgs& collArg = this->collArgs[localRank][collIdx];
        CHECK_CALL(collArg.SetArgs(globalRank, this->totalRanks,
                                   this->deviceIds[localRank],
                                   funcType, dataType,
                                   numInputElements, numOutputElements,
                                   options));
        if (this->verbose) INFO("Rank %d on child %d sets collective %d [%s]\n",
                                globalRank, this->childId, collIdx,
                                collArg.GetDescription().c_str());

        // If pre-mult scalars are provided, then create a custom reduction operator
        if (options.scalarMode >= 0)
        {
          CHILD_NCCL_CALL(ncclRedOpCreatePreMulSum(&collArg.options.redOp,
                                                   collArg.localScalar.ptr,
                                                   dataType,
                                                   (ncclScalarResidence_t)options.scalarMode,
                                                   this->comms[localRank]),
                          "ncclRedOpCreatePreMulSum");
          if (verbose) INFO("Child %d created custom redop %d for collective %d\n",
                            this->childId, collArg.options.redOp, collIdx);
        }
      }
    }
    if (this->verbose) INFO("Child %d finishes SetCollectiveArgs()\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::AllocateMem()
  {
    if (this->verbose) INFO("Child %d begins AllocateMem()\n", this->childId);

    // Read values sent by parent [see TestBed::AllocateMem()]
    int    globalRank;
    int    collId;
    bool   inPlace;
    bool   useManagedMem;

    PIPE_READ(globalRank);
    PIPE_READ(collId);
    PIPE_READ(inPlace);
    PIPE_READ(useManagedMem);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      ERROR("Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

    for (int collIdx = 0; collIdx < collArgs[localRank].size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        CollectiveArgs& collArg = this->collArgs[localRank][collIdx];
        CHECK_CALL(collArg.AllocateMem(inPlace, useManagedMem));
        if (this->verbose) INFO("Rank %d on child %d allocates memory for collective %d on device %d (%s,%s) Input: %p Output %p\n",
                                globalRank, this->childId, collIdx, this->deviceIds[localRank],
                                inPlace ? "in-place" : "out-of-place",
                                useManagedMem ? "managed" : "unmanaged",
                                collArg.inputGpu.ptr,
                                collArg.outputGpu.ptr);
      }
    }

    if (this->verbose) INFO("Child %d finishes AllocateMem()\n", this->childId);
    return TEST_SUCCESS;
  }

  // Fill input memory with pre-known patterned based on rank
  ErrCode TestBedChild::PrepareData()
  {
    if (this->verbose) INFO("Child %d begins PrepareData()\n", this->childId);

    // Read values sent by parent [see TestBed::PrepareData()]
    int globalRank;
    int collId;
    CollFuncPtr prepDataFunc;

    PIPE_READ(globalRank);
    PIPE_READ(collId);
    PIPE_READ(prepDataFunc);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      ERROR("Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }

    int const localRank = globalRank - rankOffset;
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

    for (int collIdx = 0; collIdx < collArgs[localRank].size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        if (this->verbose) INFO("Rank %d on child %d prepares data for collective %d\n",
                                globalRank, this->childId, collIdx);
        CHECK_CALL(this->collArgs[localRank][collIdx].PrepareData(prepDataFunc));
      }
    }
    if (this->verbose) INFO("Child %d finishes PrepareData()\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::ExecuteCollectives()
  {
    bool useHipGraph = false;
    PIPE_READ(useHipGraph);

    int numRanksToExecute, tempRank;
    std::vector<int> ranksToExecute = {};
    PIPE_READ(numRanksToExecute);

    for (int rank = 0; rank < numRanksToExecute; ++rank){
      PIPE_READ(tempRank);
      ranksToExecute.push_back(tempRank - this->rankOffset);
    }
    if (this->verbose) INFO("Child %d begins ExecuteCollectives() %s\n", this->childId, useHipGraph ? "(using hipGraphs)" : "");

    // Determine which local ranks to execute on
    std::vector<int> localRanksToExecute;
    for (int localRank = 0; localRank < this->deviceIds.size(); ++localRank)
    {
      // If ranksToExeute is empty, execute all local ranks belonging to this child
      if (!ranksToExecute.empty() &&
          (std::count(ranksToExecute.begin(), ranksToExecute.end(), localRank) == 0)) continue;
      localRanksToExecute.push_back(localRank);
    }

    numRanksToExecute = (int)localRanksToExecute.size();
    hipGraph_t graphs[numRanksToExecute];
    hipGraphExec_t graphExec[numRanksToExecute];

    // Start HIP graph stream capture if requested
    if (useHipGraph)
    {
      for (int localRank : localRanksToExecute)
      {
        if (this->verbose) INFO("Capturing stream for rank %d\n", localRank);
        CHECK_HIP(hipStreamBeginCapture(this->streams[localRank], hipStreamCaptureModeGlobal));
      }
    }

    // Start group call
    CHILD_NCCL_CALL(ncclGroupStart(), "ncclGroupStart");

    // Loop over all collectives to be executed in group call
    for (int collId = 0; collId < this->numCollectivesInGroup; ++collId)
    {
      // Loop over all local ranks
      for (int localRank : localRanksToExecute)
      {
        CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

        CollectiveArgs const& collArg = this->collArgs[localRank][collId];

        if (this->printValues && !useHipGraph)
        {
          int const numInputElementsToPrint = (this->printValues < 0 ? collArg.numInputElements : this->printValues);
          PtrUnion inputCpu;
          size_t const numInputBytes = numInputElementsToPrint * DataTypeToBytes(collArg.dataType);
          inputCpu.AllocateCpuMem(numInputBytes);
          CHECK_HIP(hipMemcpy(inputCpu.ptr, collArg.inputGpu.ptr, numInputBytes, hipMemcpyDeviceToHost));
          printf("[ DEBUG    ] Rank %02d Coll %d %-10s: %s\n", collArg.globalRank, collId, "Input",
                 inputCpu.ToString(collArg.dataType, numInputElementsToPrint).c_str());
          inputCpu.FreeCpuMem();

          int const numOutputElementsToPrint = (this->printValues < 0 ? collArg.numOutputElements : this->printValues);
          size_t const numOutputBytes = numOutputElementsToPrint * DataTypeToBytes(collArg.dataType);
          CHECK_HIP(hipMemcpy(collArg.outputCpu.ptr, collArg.outputGpu.ptr, numOutputBytes, hipMemcpyDeviceToHost));
          printf("[ DEBUG    ] Rank %02d Coll %d %-10s: %s\n", collArg.globalRank, collId, "Pre-Output",
                 collArg.outputCpu.ToString(collArg.dataType, numOutputElementsToPrint).c_str());
        }

        switch (collArg.funcType)
        {
        case ncclCollBroadcast:
          CHILD_NCCL_CALL(ncclBroadcast(collArg.inputGpu.ptr,
                                        collArg.outputGpu.ptr,
                                        collArg.numInputElements,
                                        collArg.dataType,
                                        collArg.options.root,
                                        this->comms[localRank],
                                        this->streams[localRank]),
                          "ncclBroadcast");
          break;
        case ncclCollReduce:
          CHILD_NCCL_CALL(ncclReduce(collArg.inputGpu.ptr,
                                     collArg.outputGpu.ptr,
                                     collArg.numInputElements,
                                     collArg.dataType,
                                     collArg.options.redOp,
                                     collArg.options.root,
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
                                            collArg.options.redOp,
                                            this->comms[localRank],
                                            this->streams[localRank]),
                          "ncclReduceScatter");
          break;
        case ncclCollAllReduce:
          CHILD_NCCL_CALL(ncclAllReduce(collArg.inputGpu.ptr,
                                        collArg.outputGpu.ptr,
                                        collArg.numInputElements,
                                        collArg.dataType,
                                        collArg.options.redOp,
                                        this->comms[localRank],
                                        this->streams[localRank]),
                          "ncclAllReduce");
          break;
        case ncclCollGather:
          CHILD_NCCL_CALL(ncclGather(collArg.inputGpu.ptr,
                                     collArg.outputGpu.ptr,
                                     collArg.numInputElements,
                                     collArg.dataType,
                                     collArg.options.root,
                                     this->comms[localRank],
                                     this->streams[localRank]),
                          "ncclGather");
          break;
        case ncclCollScatter:
          CHILD_NCCL_CALL(ncclScatter(collArg.inputGpu.ptr,
                                      collArg.outputGpu.ptr,
                                      collArg.numOutputElements,
                                      collArg.dataType,
                                      collArg.options.root,
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
        case ncclCollAllToAllv:
          CHILD_NCCL_CALL(ncclAllToAllv(collArg.inputGpu.ptr,
                                        collArg.options.sendcounts + (this->rankOffset + localRank)*this->totalRanks,
                                        collArg.options.sdispls + (this->rankOffset + localRank)*this->totalRanks,
                                        collArg.outputGpu.ptr,
                                        collArg.options.recvcounts + (this->rankOffset + localRank)*this->totalRanks,
                                        collArg.options.rdispls + (this->rankOffset + localRank)*this->totalRanks,
                                        collArg.dataType,
                                        this->comms[localRank],
                                        this->streams[localRank]),
                          "ncclAllToAllv");
          break;
        case ncclCollSend:
          CHILD_NCCL_CALL(ncclSend(collArg.inputGpu.ptr,
                                   collArg.numInputElements,
                                   collArg.dataType,
                                   collArg.options.root,
                                   this->comms[localRank],
                                   this->streams[localRank]),
                          "ncclSend");
          break;
        case ncclCollRecv:
          CHILD_NCCL_CALL(ncclRecv(collArg.outputGpu.ptr,
                                   collArg.numOutputElements,
                                   collArg.dataType,
                                   collArg.options.root,
                                   this->comms[localRank],
                                   this->streams[localRank]),
                          "ncclRecv");
          break;
        default:
          ERROR("Unknown func type %d\n", collArg.funcType);
          return TEST_FAIL;
        }
      }
    }

    // End group call
    CHILD_NCCL_CALL(ncclGroupEnd(), "ncclGroupEnd");

    // Instantiate and launch HIP graph if requested
    if (useHipGraph)
    {
      for (int localRank : localRanksToExecute)
      {
        if (this->verbose) INFO("Ending stream capture for rank %d\n", localRank);

        CHECK_HIP(hipStreamEndCapture(this->streams[localRank], &graphs[localRank]));
        if (this->verbose)
        {
          size_t numNodes;
          hipGraphNode_t* nodes;
          CHECK_HIP(hipGraphGetNodes(graphs[localRank], nodes, &numNodes));
          INFO("Graph for rank %d has %lu nodes\n", localRank, numNodes);
        }

        if (this->verbose) INFO("Instantiating executable graph for rank %d\n", localRank);
        CHECK_HIP(hipGraphInstantiate(&graphExec[localRank], graphs[localRank], NULL, NULL, 0));
      }

      for (int localRank : localRanksToExecute)
      {
        if (this->verbose) INFO("Launch graph for rank %d\n", localRank);
        CHECK_HIP(hipGraphLaunch(graphExec[localRank], this->streams[localRank]));
      }
    }
    else
    {
      if (this->verbose)
        INFO("Child %d submits group call.  Waiting for completion\n", this->childId);
    }

    // Synchronize
    for (int localRank : localRanksToExecute)
    {
      if (this->verbose) INFO("Starting synchronization for rank %d\n", localRank);
      CHECK_HIP(hipStreamSynchronize(this->streams[localRank]));
    }

    // Destroy graphs
    if (useHipGraph)
    {
      for (int localRank : localRanksToExecute)
      {
        if (this->verbose) INFO("Destroying graphs for rank %d\n", localRank);
        CHECK_HIP(hipGraphDestroy(graphs[localRank]));
        CHECK_HIP(hipGraphExecDestroy(graphExec[localRank]));
      }
    }

    if (this->printValues)
    {
      for (int collId = 0; collId < this->numCollectivesInGroup; ++collId)
        for (int localRank : localRanksToExecute)
        {
          CollectiveArgs const& collArg = this->collArgs[localRank][collId];

          int numOutputElementsToPrint = (this->printValues < 0 ? collArg.numOutputElements : this->printValues);
          size_t const numOutputBytes = numOutputElementsToPrint * DataTypeToBytes(collArg.dataType);
          CHECK_HIP(hipMemcpy(collArg.outputCpu.ptr, collArg.outputGpu.ptr, numOutputBytes, hipMemcpyDeviceToHost));
          printf("[ DEBUG    ] Rank %02d Coll %d %-10s: %s\n", collArg.globalRank, collId, "Output",
                 collArg.outputCpu.ToString(collArg.dataType, numOutputElementsToPrint).c_str());

          printf("[ DEBUG    ] Rank %02d Coll %d %-10s: %s\n", collArg.globalRank, collId, "Expected",
                 collArg.expected.ToString(collArg.dataType, numOutputElementsToPrint).c_str());
        }
    }
    if (this->verbose) INFO("Child %d finishes ExecuteCollectives()\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::ValidateResults()
  {
    // Read values sent by parent [see TestBed::ValidateResults()]
    int globalRank, collId;
    PIPE_READ(globalRank);
    PIPE_READ(collId);

    if (this->verbose) INFO("Child %d begins ValidateResults()\n", this->childId);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      ERROR("Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

    ErrCode status = TEST_SUCCESS;
    for (int collIdx = 0; collIdx < collArgs[localRank].size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        if (this->verbose) INFO("Rank %d on child %d validating collective %d results\n",
                                globalRank, this->childId, collIdx);
        if (this->collArgs[localRank][collIdx].ValidateResults() != TEST_SUCCESS)
        {
          ERROR("Rank %d Collective %d output does not match expected\n", globalRank, collIdx);
          status = TEST_FAIL;
        }
      }
    }
    if (this->verbose) INFO("Child %d finishes ValidateResults() with status %s\n", this->childId,
                            status == TEST_SUCCESS ? "SUCCESS" : "FAIL");
    return status;
  }

  ErrCode TestBedChild::DeallocateMem()
  {
    if (this->verbose) INFO("Child %d begins DeallocateMem\n", this->childId);

    // Read values sent by parent [see TestBed::DeallocateMem()]
    int globalRank, collId;
    PIPE_READ(globalRank);
    PIPE_READ(collId);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      ERROR("Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

    for (int collIdx = 0; collIdx < collArgs[localRank].size(); ++collIdx)
    {
      CollectiveArgs& collArg = this->collArgs[localRank][collIdx];
      if (collId == -1 || collId == collIdx)
      {
        if (this->verbose)
        {
          INFO("Child %d release memory for collective %d (Input: %p Output %p\n",
               this->childId, collIdx, collArg.inputGpu.ptr, collArg.outputGpu.ptr);
        }

        CHECK_CALL(collArg.DeallocateMem());
      }
      if (collArg.options.scalarMode != -1)
      {
        CHILD_NCCL_CALL(ncclRedOpDestroy(collArg.options.redOp, this->comms[localRank]),
                        "ncclRedOpDestroy");
        if (verbose) INFO("Child %d destroys custom redop %d for collective %d\n",
                          this->childId, collArg.options.redOp, collIdx);
      }
    }
    if (this->verbose) INFO("Child %d finishes DeallocateMem\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::DestroyComms()
  {
    if (this->verbose) INFO("Child %d begins DestroyComms\n", this->childId);

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
    this->streams.clear();
    if (this->verbose) INFO("Child %d finishes DestroyComms\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::Stop()
  {
    return TEST_SUCCESS;
  }
}
