/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <mpi.h>
#include <fstream>
#include <unordered_set>

#include "rcclReplayer.hpp"

#include <dirent.h>
#include <stdio.h>

using namespace rccl;

static int json_format = 0; // binary by default

// move to inside class or kept as static var
static constexpr size_t rcclCallSize = sizeof(rcclApiCall);
static char line[rcclCallSize]; // size of collectivecall struct
static int lineNum = 0;
static ncclUniqueId uniqueId;

// assuming shared file system or similar
// should this be replayer or in main
static int ParseLogFormat(const char* logFormat, std::string& filename, std::string& extension)
{
  int json_format = 0;
  size_t dot;
  if ((dot = std::string(logFormat).find(".")) != std::string::npos)
  {
    filename = std::string(logFormat).substr(0, dot);
    extension = std::string(logFormat).substr(dot);
    if (extension.compare(".json") == 0)
    {
      json_format = 1;
    }
  } else {
    filename = std::string(logFormat);
  }
  return json_format;
  // TODO: modularize and reuse this snippet from recorder
}

Replayer::Replayer(const std::string& logname, int json_format, int rank, int size) : myRank(rank),
                                                                                      numGlobalRanks(size)
{
  log.open(logname, json_format ? std::ifstream::in : std::ifstream::binary);
}

void Replayer::parse()
{
  while (log.read(line, rcclCallSize)) // istream::get fail here when running into newline
  {
    rcclApiCall call = *((rcclApiCall*) line);

    if (call.sendPtrBase)
    {
      if (!dMemMap.contains(call.sendPtrBase))
      {
        dMemMap[call.sendPtrBase].size = call.sendPtrExtent;
      }
      dMemMap[call.sendPtrBase].lastLineUsed = lineNum;
    }
    if (call.recvPtrBase)
    {
      if (!dMemMap.contains(call.recvPtrBase))
      {
        dMemMap[call.recvPtrBase].size = call.recvPtrExtent;
      }
      dMemMap[call.recvPtrBase].lastLineUsed = lineNum;
    }
    if (call.stream)
    {
      // Only set device on first use (when stream will be created)
      if (streams[call.stream].lastLine < 0) {
        streams[call.stream].device = call.hipDev;
      }
      streams[call.stream].lastLine = lineNum;
    }

    switch (call.type) {
    case rrGroupStart:
    case rrGroupEnd:
    case rrGroupSimulatedEnd: // TODO
    case rrCommInitRank:
    /// case rrCommInitRankConfig:   <-- these all should depend on CommInitDev
    case rrCommSplit: // <-- not covered for now dealt with in replay time
    case rrCommFinalize:
    case rrCommDestroy:
    case rrCommAbort:
    case rrCommRegister:
    case rrCommDeregister: // I think commDeregister is not affected by handle in both way?
    case rrMemFree:
    case rrRedOpCreatePreMulSum:
    case rrRedOpDestroy:
    case rrOtherCall:
    {
      break; // no op
    }
  // Communicator
    case rrGetUniqueId:
    {
      idRankMap[call.commId];
      break;
    }
    
    case rrCommInitDev:             // which should capture all comm - uniqueID relations
    {
      Ids.push_back(call.commId);
      // for debugging might want a reverse map
      break;
    }
    case rrCommInitAll:
    {
      if (call.sendbuff)
      {
        log.ignore(call.root * sizeof(int));
      }
      break;
    }

  // Memory allocation
    //integrate these later
    case rrMemAlloc:
    {
      // Replayer will not free this without explicit ncclMemFree
      dMemMap[call.recvbuff].size = call.count;
      break;
    }

    case rrAllToAllv:
    {
      // For multi-threaded replay, we need to pre-read AllToAllv data
      // Read and store the variable-length data, then store with ThreadOp below
        break;
    }
    default: // collectives
    {
      /*  if capturing:
       *    if first time (start.empty)
       *      init stream
       *      push this line for replayer later
       *    increment depth
       *  else
       *    use internal counter to separate diff graph launch
       */
      if (call.graphCaptured == 1)
      {
        if (!graphLife.contains(call.graphID))
        {
          graphLife[call.graphID].starts.insert(lineNum);
          graphLife[call.graphID].stream = call.stream;
        }
        graphLife[call.graphID].depth++;
        graphLife[call.graphID].counter++;
        graphLife[call.graphID].end = lineNum; // in case the graph never gets launched
      } else if (call.graphID) {
        if (graphLife[call.graphID].counter == graphLife[call.graphID].depth)
        {
          graphLife[call.graphID].starts.insert(lineNum);
        }
        graphLife[call.graphID].counter--;
        if (graphLife[call.graphID].counter == 0)
        {
          graphLife[call.graphID].end = lineNum; // we currently sync graph after its last launch
                                                 // for convenience of graph destroy, may later
                                                 // need a comm->graphs map so that CommReclaim dont hang
          graphLife[call.graphID].counter = graphLife[call.graphID].depth;
        }
      }
    }
    }
    
    // Store each operation grouped by its originating thread ID
    ThreadOp op = {call, lineNum, {}, {}, {}, {}};
    
    // For AllToAllv, read and store the variable-length count/displacement arrays
    if (call.type == rrAllToAllv) {
      int size = call.nRanks;
      op.sendcounts.resize(size);
      op.sdispls.resize(size);
      op.recvcounts.resize(size);
      op.rdispls.resize(size);
      log.read((char*)op.sendcounts.data(), size * sizeof(size_t));
      log.read((char*)op.sdispls.data(), size * sizeof(size_t));
      log.read((char*)op.recvcounts.data(), size * sizeof(size_t));
      log.read((char*)op.rdispls.data(), size * sizeof(size_t));
    }
    
    threadOps[call.tid].push_back(std::move(op));
    
    lineNum++;
  }

  // Extract unique thread IDs from the map keys
  for (auto& [tid, ops] : threadOps) {
    uniqueThreads.push_back(tid);
  }
  // Sort for deterministic replay order
  std::sort(uniqueThreads.begin(), uniqueThreads.end());
  
  printf("[INFO    ] Rank %d : Found %zu unique threads in log\n", myRank, uniqueThreads.size());

  // exchange communicator info
  std::vector<int> comm_count(numGlobalRanks);
  comm_count[myRank] = Ids.size();
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, comm_count.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> displs(comm_count.size() + 1, 0);
  std::inclusive_scan(comm_count.begin(), comm_count.end(), displs.begin() + 1);
  int aggragatedCommCount = std::reduce(comm_count.begin(), comm_count.end());
  /*
   *                  rank1, comm_count[1]xID  r2, comm_count[2]  r3 ...  r4 ...
   *  AllRankCommIds [------------------------+-----------------+-------+---------+....]
   */
  std::vector<uint64_t> AllRankCommIds(aggragatedCommCount);
  MPI_Allgatherv(Ids.data(), Ids.size(), MPI_UINT64_T,
                 AllRankCommIds.data(), comm_count.data(), displs.data(), MPI_UINT64_T, MPI_COMM_WORLD);

  int k = 0;
  for (int i = 0; i < numGlobalRanks; i++)
  {
    if (i == myRank)
    {
      k += Ids.size();
      continue;
    }
    for (int j = 0; j < comm_count[i]; j++)
    {
      if (idRankMap.contains(AllRankCommIds[k]))
      {
        idRankMap[AllRankCommIds[k]].push_back(i);
      }
      k++;
    }
  }

  lineNum = 0;
  log.clear();
  log.seekg(0, std::ios_base::beg);
  // TODO: print out resources here allocated if requested
}

void Replayer::replay()
{
  // Use multi-threaded replay when multiple threads detected
  if (uniqueThreads.size() > 1) {
    printf("[INFO    ] Rank %d : Multi-threaded replay with %zu threads\n", myRank, uniqueThreads.size());
    replayMultiThreaded();
    return;
  }
  
  printf("[INFO    ] Rank %d : Single-threaded replay\n", myRank);
  
  // Single-threaded replay (original path)
  while (log.read(line, rcclCallSize))
  {
    rcclApiCall call = *((rcclApiCall*) line);
    printf("[INFO    ] Rank %d - Line %d : %s\n", myRank, lineNum, rcclCallStr[call.type]);
    HIP_CALL(hipSetDevice(call.hipDev));
    void *sbuffer = NULL, *rbuffer = NULL;

    if (call.type < rrGroupStart)
    {
      if ((call.sendPtrBase && !dMemMap.contains(call.sendPtrBase)) || (call.recvPtrBase && !dMemMap.contains(call.recvPtrBase)))
      {
        printf("[ERROR   ] Rank %d - Line %d : Unknown buffer in collectives\n", myRank, lineNum);
        exit(1);
      }

      if (call.sendPtrBase)
      {
        if (!dMemMap[call.sendPtrBase].base)
        {
              HIP_CALL(hipMalloc(&dMemMap[call.sendPtrBase].base, dMemMap[call.sendPtrBase].size));
        }
        std::ptrdiff_t diff = (char*)call.sendbuff - (char*)call.sendPtrBase;
        sbuffer = (char*)dMemMap[call.sendPtrBase].base + diff;
      }
      if (call.recvPtrBase)
      {
        if (!dMemMap[call.recvPtrBase].base)
        {
             HIP_CALL(hipMalloc(&dMemMap[call.recvPtrBase].base, dMemMap[call.recvPtrBase].size));
        }
        std::ptrdiff_t diff = (char*)call.recvbuff - (char*)call.recvPtrBase;
        rbuffer = (char*)dMemMap[call.recvPtrBase].base + diff;
      }

      //stream
      if (call.stream && !streams[call.stream].stream)
      {
        HIP_CALL(hipStreamCreate(&streams[call.stream].stream));
        streams[call.stream].device = call.hipDev;  // Track device when creating stream
      }

      //graph
      /*
       *  if capturing
       *    if firstime (line in start)
       *      stream capture begin
       *    else if stream differ from initial capturing stream
       *      //create dependency
       *    if depth reached // after call execution switch
       *      conclude graph
       *  else (launching)
       */
      if (call.graphCaptured == 1)
      {
        graphLife[call.graphID].counter--;
        if (graphLife[call.graphID].starts.contains(lineNum))
        {
          HIP_CALL(hipStreamBeginCapture(streams[call.stream].stream, hipStreamCaptureModeRelaxed));
          printf("[INFO    ] Rank %d - Line %d : starting capture graph %llu\n", myRank, lineNum, call.graphID);
        } else if (graphLife[call.graphID].stream != call.stream) {
          printf("[WARNING ] \x1b[31mRank %d - Line %d : multi-stream graph may not replay original dependency accurately\x1b[0m\n", myRank, lineNum);
        hipEvent_t event;
        HIP_CALL(hipEventCreate(&event));
          graphLife[call.graphID].events.push_back(event);
          HIP_CALL(hipEventRecord(event, streams[graphLife[call.graphID].stream].stream));
          HIP_CALL(hipStreamWaitEvent(streams[call.stream].stream, event));
        }    
      } else if (call.graphID) {
        if (graphLife[call.graphID].starts.contains(lineNum))
        {
          printf("[INFO    ] Rank %d - Line %d : launching graph %llu\n", myRank, lineNum, call.graphID);
          HIP_CALL(hipGraphLaunch(graphLife[call.graphID].graphExec, streams[call.stream].stream));
        }
        printf("[INFO    ] Rank %d - Line %d : being played by previous graph %llu\n", myRank, lineNum, call.graphID);
        goto cleanup;
      }
    }

    switch (call.type) {
    case rrGroupSimulatedEnd: // TODO: cannot test atm
    /// case rrCommInitRankConfig:   <-- these all should depend on CommInitDev
    case rrRedOpCreatePreMulSum:
    case rrRedOpDestroy:
    case rrOtherCall:
    {
      printf("[ERROR   ] Rank %d - Line %d : Unexpected call: %s\n", myRank, lineNum, rcclCallStr[call.type]);
      exit(1);
    }

    // To be integrated later
    case rrCommFinalize:
    {
      NCCL_CALL(ncclCommFinalize(commMap[call.comm]));
      break;
    }
    case rrCommDestroy:
    {
      NCCL_CALL(ncclCommDestroy(commMap[call.comm]));
      break;
    }
    case rrCommAbort:
    {
      NCCL_CALL(ncclCommAbort(commMap[call.comm]));
      break;
    }

    case rrGroupStart:
    {
      NCCL_CALL(ncclGroupStart());
      break;
    }
    case rrGroupEnd:
    {
      NCCL_CALL(ncclGroupEnd());
      break;
    }

    case rrGetUniqueId:
    {
      NCCL_CALL(ncclGetUniqueId(&uniqueId));
      idMap[call.commId] = uniqueId;
      break;
    }
    case rrCommInitRank:
    {
      lastCall = rrCommInitRank;
      break;
    }
    /// case rrCommInitRankConfig:
    case rrCommInitDev:
    {
      if (lastCall == rrCommInitAll) // no other calls between ncclCommInitAll and ncclCommInitRankDev
      {                              // nor ncclCommInitRankDev not proceeded by ncclCommInitAll/Rank()
        goto cleanup;
      }
      // set device
      // TODO: double check this, since some version of NCCL theres a reset to original device
      HIP_CALL(hipSetDevice(call.root));

      if (!idMap.contains(call.commId))
      {
        MPI_Recv(&uniqueId, sizeof(ncclUniqueId), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else {
        for (int rank : idRankMap[call.commId])
        {
          MPI_Send(&idMap[call.commId], sizeof(ncclUniqueId), MPI_BYTE, rank, 0, MPI_COMM_WORLD);
        }
        uniqueId = idMap[call.commId]; // <- double check in case of bug/communicator init hang
      }
      ncclComm_t comm;
      NCCL_CALL(ncclCommInitRank(&comm, call.nRanks, uniqueId, call.globalRank));
      commMap[call.comm] = comm;
      break;
    }
    case rrCommInitAll:
    {
      int ndev = call.root;
      int *devlist = NULL;
      if (call.sendbuff)
      {
        std::vector<int> devices(ndev);
        log.read((char*)devices.data(), ndev * sizeof(int));
        devlist = devices.data();
      }
      ncclComm_t comm;
      NCCL_CALL(ncclCommInitAll(&comm, ndev, devlist));
      commMap[call.comm] = comm;
      break;
    }
    case rrCommSplit:
    {
      int color = call.nRanks;
      int key = call.globalRank;
      // TODO: parse config later
      ncclComm_t newcomm;
      ncclComm_t comm = (ncclComm_t) call.commId;
      NCCL_CALL(ncclCommSplit(commMap[comm], color, key, &newcomm, NULL));
      commMap[call.comm/*original newcomm to commSplit call*/] = newcomm;
      break;
    }


    case rrCommRegister:
    {
      if (!dMemMap.contains(call.sendPtrBase) || !commMap.contains(call.comm))
      {
        printf("[ERROR   ] Rank %d - Line %d : Unknown buffer for CommRegister\n", myRank, lineNum);
        exit(1);
      }
      if (!dMemMap[call.sendPtrBase].base)
      {
        HIP_CALL(hipMalloc(&dMemMap[call.sendPtrBase].base, dMemMap[call.sendPtrBase].size));
      }
      sbuffer = (char*)dMemMap[call.sendPtrBase].base + (std::ptrdiff_t)((char*)call.sendbuff - (char*)call.sendPtrBase);
      NCCL_CALL(ncclCommRegister(commMap[call.comm], sbuffer, dMemMap[call.sendPtrBase].size, &handleMap[call.recvbuff]));
      break;
    }
    case rrCommDeregister:
    {
      NCCL_CALL(ncclCommDeregister(commMap[call.comm], handleMap[call.recvbuff]));
      break;
    }
    case rrMemAlloc:
    {
      NCCL_CALL(ncclMemAlloc(&dMemMap[call.recvbuff].base, call.count));
      break ;
    }
    case rrMemFree:
    {
      NCCL_CALL(ncclMemFree(dMemMap[call.recvbuff].base));
      break;
    }

    // TODO: further simplify switch base on common parameters
    // no op or root
    case rrAllToAll:
    {
      NCCL_CALL(ncclAllToAll(sbuffer, rbuffer, call.count, call.datatype, commMap[call.comm], streams[call.stream].stream));
      break;
    }
    case rrAllGather:
    {
      NCCL_CALL(ncclAllGather(sbuffer, rbuffer, call.count, call.datatype, commMap[call.comm], streams[call.stream].stream));
      break;
    }
    // op root
    case rrReduce:
    {
      NCCL_CALL(ncclReduce(sbuffer, rbuffer, call.count, call.datatype, call.op, call.root, commMap[call.comm], streams[call.stream].stream));
      break;
    }
    // root
    case rrBroadcast:
    {
      NCCL_CALL(ncclBroadcast(sbuffer, rbuffer, call.count, call.datatype, call.root, commMap[call.comm], streams[call.stream].stream));
      break;
    }
    case rrScatter:
    {
      NCCL_CALL(ncclScatter(sbuffer, rbuffer, call.count, call.datatype, call.root, commMap[call.comm], streams[call.stream].stream));
      break;
    }
    case rrGather:
    {
      NCCL_CALL(ncclGather(sbuffer, rbuffer, call.count, call.datatype, call.root, commMap[call.comm], streams[call.stream].stream));
      break;
    }
    // root -
    case rrBcast:
    {
      NCCL_CALL(ncclBcast(rbuffer, call.count, call.datatype, call.root, commMap[call.comm], streams[call.stream].stream));
      break;
    }
    case rrSend:
    {
      NCCL_CALL(ncclSend(rbuffer, call.count, call.datatype, call.root, commMap[call.comm], streams[call.stream].stream));
      break;
    }
    case rrRecv:
    {
      NCCL_CALL(ncclRecv(rbuffer, call.count, call.datatype, call.root, commMap[call.comm], streams[call.stream].stream));
      break;
    }
    // op
    case rrReduceScatter:
    {
      NCCL_CALL(ncclReduceScatter(sbuffer, rbuffer, call.count, call.datatype, call.op, commMap[call.comm], streams[call.stream].stream));
      break;
    }
    case rrAllReduce:
    {
      NCCL_CALL(ncclAllReduce(sbuffer, rbuffer, call.count, call.datatype, call.op, commMap[call.comm], streams[call.stream].stream));
      break;
    }
    case rrAllReduceWithBias:
    {
      std::vector<char> acc(call.count * ncclTypeSize(call.datatype));
      NCCL_CALL(ncclAllReduceWithBias(sbuffer, rbuffer, call.count, call.datatype, call.op, commMap[call.comm], streams[call.stream].stream, acc.data()));
      HIP_CALL(hipStreamSynchronize(streams[call.stream].stream)); // TODO: remove, and further verify behavior of fused AR
      break;
    }
    // a2av
    case rrAllToAllv:
    {
      // timer pause here
      // assuming blocking for now
      int size = call.nRanks;
      std::vector<size_t> sendcounts(size), sdispls(size), recvcounts(size), rdispls(size);
      log.read((char*)sendcounts.data(), size * sizeof(size_t));
      log.read((char*)sdispls.data(), size * sizeof(size_t));
      log.read((char*)recvcounts.data(), size * sizeof(size_t));
      log.read((char*)rdispls.data(), size * sizeof(size_t));
      
      NCCL_CALL(ncclAllToAllv(sbuffer, sendcounts.data(), sdispls.data(), rbuffer, recvcounts.data(), rdispls.data(),
                              call.datatype, commMap[call.comm], streams[call.stream].stream));
      HIP_CALL(hipStreamSynchronize(streams[call.stream].stream)); // TODO: remove
      break;
    }
    } //switch
    printf("[INFO    ] Rank %d - Line %d : %s called\n", myRank, lineNum, rcclCallStr[call.type]);
    lastCall = call.type;

    if (call.graphCaptured == 1)
    {
      // TODO: This requires further testing
      if (graphLife[call.graphID].stream != call.stream)
      {
        hipEvent_t event;
        HIP_CALL(hipEventCreate(&event));
        graphLife[call.graphID].events.push_back(event);
        HIP_CALL(hipEventRecord(event, streams[call.stream].stream));
        HIP_CALL(hipStreamWaitEvent(streams[graphLife[call.graphID].stream].stream, event));
      }
      if (graphLife[call.graphID].counter == 0)
      {
	hipGraphNode_t temp;
	char errbuff[3000];
        HIP_CALL(hipStreamEndCapture(streams[graphLife[call.graphID].stream].stream, &graphLife[call.graphID].graph));
        // TODO: confirm with clr behavior of graphInstantiate in face of failure
        HIP_CALL(hipGraphInstantiate(&graphLife[call.graphID].graphExec, graphLife[call.graphID].graph, &temp, errbuff, 3000));
        for (hipEvent_t e : graphLife[call.graphID].events)
        {
          HIP_CALL(hipEventDestroy(e));
        }
      }
    }

cleanup:
    printf("[INFO    ] Rank %d - Line %d : cleaning up\n", myRank, lineNum);
    
    // Free resources if possible
    if (call.sendPtrBase && lineNum == dMemMap[call.sendPtrBase].lastLineUsed) {
      // TODO: free contains a sync, may need a second thought
      //       also this may proceed commDeregister in case of UBR thus susceptible to change in implementation
      HIP_CALL(hipFree(dMemMap[call.sendPtrBase].base));
      dMemMap[call.sendPtrBase].base = NULL; // in case of in place ops
    }
    if (call.recvPtrBase && lineNum == dMemMap[call.recvPtrBase].lastLineUsed && dMemMap[call.recvPtrBase].base) {
      HIP_CALL(hipFree(dMemMap[call.recvPtrBase].base));
    }
    if (call.graphID && lineNum == graphLife[call.graphID].end) {
      HIP_CALL(hipStreamSynchronize(streams[call.stream].stream));
      HIP_CALL(hipGraphExecDestroy(graphLife[call.graphID].graphExec));
      HIP_CALL(hipGraphDestroy(graphLife[call.graphID].graph));
    }
    // Destroy stream on its last use (switch to correct device first)
    if (call.stream && lineNum == streams[call.stream].lastLine) {
      HIP_CALL(hipSetDevice(streams[call.stream].device));  // Switch to device where stream was created
      if (call.graphCaptured != 1) {
        HIP_CALL(hipStreamSynchronize(streams[call.stream].stream));  // Wait for pending ops
      }
      HIP_CALL(hipStreamDestroy(streams[call.stream].stream));
    }
    lineNum++; // change for a2av
  }
}

// Multi-threaded replay
void Replayer::replayMultiThreaded()
{
  // Containers to separate operations into 3 phases
  std::vector<ThreadOp> commInitOps;     // Phase 1: comm creation (must be sequential)
  std::vector<ThreadOp> commDestroyOps;  // Phase 3: comm destruction (after all work)
  std::map<int, std::vector<ThreadOp>> workerOps;  // Phase 2: concurrent ops per thread
  
  // Categorize all operations by type
  for (auto& [tid, ops] : threadOps) {
    for (auto& op : ops) {
      // Comm init ops must be done sequentially with MPI coordination
      if (op.call.type == rrGetUniqueId || op.call.type == rrCommInitRank || 
          op.call.type == rrCommInitDev || op.call.type == rrCommInitAll) {
        commInitOps.push_back(op);
      // Comm destroy must wait until all concurrent work is done
      } else if (op.call.type == rrCommDestroy || op.call.type == rrCommAbort) {
        commDestroyOps.push_back(op);
      // Everything else can run concurrently per-thread
      } else {
        workerOps[tid].push_back(op);
      }
    }
  }
  
  // Sort by line number to preserve original execution order
  std::sort(commInitOps.begin(), commInitOps.end(), 
            [](const ThreadOp& a, const ThreadOp& b) { return a.lineNum < b.lineNum; });
  std::sort(commDestroyOps.begin(), commDestroyOps.end(),
            [](const ThreadOp& a, const ThreadOp& b) { return a.lineNum < b.lineNum; });
  
  // ========== PHASE 1: Create communicators ==========
  std::vector<ThreadOp> commInitDevOps;
  for (auto& op : commInitOps) {
    // Collect actual comm creation ops
    if (op.call.type == rrCommInitDev) {
      commInitDevOps.push_back(op);
    // Generate unique IDs for comms this rank is responsible for
    } else if (op.call.type == rrGetUniqueId) {
      NCCL_CALL(ncclGetUniqueId(&uniqueID));
      idMap[op.call.commId] = uniqueID;  // Store mapping: logged_id -> new_id
    }
  }
  
  // Sort CommInitDev ops by uniqueId (commId) to ensure all ranks process in same order
  // Line numbers differ across ranks, but uniqueIds are the same!
  std::sort(commInitDevOps.begin(), commInitDevOps.end(),
            [](const ThreadOp& a, const ThreadOp& b) { return a.call.commId < b.call.commId; });
  
  printf("[INFO    ] Rank %d : Phase 1 - Creating %zu communicators\n", myRank, commInitDevOps.size());
  
  // Create each communicator with MPI coordination
  for (auto& op : commInitDevOps) {
    ncclUniqueId id;
    // Check if this rank has the unique ID for this comm
    int hasId = idMap.contains(op.call.commId) ? 1 : 0;
    int root = -1;
    
    // Gather from all ranks: who has this unique ID
    std::vector<int> hasIdVec(numGlobalRanks);
    MPI_Allgather(&hasId, 1, MPI_INT, hasIdVec.data(), 1, MPI_INT, MPI_COMM_WORLD);
    // Find the rank that will broadcast the unique ID
    for (int i = 0; i < numGlobalRanks; i++) {
      if (hasIdVec[i]) { root = i; break; }
    }
    
    if (root < 0) {
      printf("[ERROR   ] Rank %d : No rank has uniqueID for comm %lu\n", myRank, op.call.commId);
      exit(1);
    }
    
    // Root copies its unique ID, then broadcasts to all ranks
    if (myRank == root) {
      id = idMap[op.call.commId];
    }
    MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, root, MPI_COMM_WORLD);
    
    // Set GPU device for this communicator
    HIP_CALL(hipSetDevice(op.call.root));
    // All ranks create the comm together (collective operation)
    // Use myRank as globalRank - log files may be assigned to different MPI ranks
    // than they were originally generated on (based on hostname matching)
    ncclComm_t comm;
    NCCL_CALL(ncclCommInitRank(&comm, op.call.nRanks, id, myRank));
    // Store mapping: logged_comm_ptr -> new_comm
    commMap[op.call.comm] = comm;
    
    printf("[INFO    ] Rank %d : Created comm %p (globalRank %d)\n", myRank, comm, myRank);
  }
  
  // Wait for all ranks to finish creating communicators
  MPI_Barrier(MPI_COMM_WORLD);
  
  // ========== PHASE 2: Concurrent operations (multi-threaded) ==========
  printf("[INFO    ] Rank %d : Phase 2 - Spawning %zu worker threads\n", myRank, workerOps.size());
  
  // Spawn one C++ thread for each original thread in the log
  std::vector<std::thread> workers;
  for (auto& [tid, ops] : workerOps) {
    workers.emplace_back(&Replayer::threadWorker, this, tid, std::ref(ops));
  }
  
  // Wait for all worker threads to complete
  for (auto& w : workers) {
    w.join();
  }
  
  // Synchronize all MPI ranks before cleanup
  MPI_Barrier(MPI_COMM_WORLD);
  
  // ========== PHASE 3: Destroy communicators ==========
  printf("[INFO    ] Rank %d : Phase 3 - Destroying %zu communicators\n", myRank, commDestroyOps.size());
  
  for (auto& op : commDestroyOps) {
    // Look up the actual comm and destroy it
    if (commMap.contains(op.call.comm)) {
      NCCL_CALL(ncclCommDestroy(commMap[op.call.comm]));
      printf("[INFO    ] Rank %d : Destroyed comm %p\n", myRank, (void*)op.call.comm);
    }
  }
  
  printf("[INFO    ] Rank %d : Multi-threaded replay complete\n", myRank);
}

// Worker function that runs in each spawned thread
void Replayer::threadWorker(int tid, const std::vector<ThreadOp>& ops)
{
  printf("[INFO    ] Rank %d Thread %d : Starting with %zu operations\n", myRank, tid, ops.size());
  
  // Process each operation assigned to this thread
  for (const auto& op : ops) {
    executeOp(op, true);  // true = use locks for thread safety
  }
  
  printf("[INFO    ] Rank %d Thread %d : Complete\n", myRank, tid);
}

// Execute a single operation (thread-safe when useLock=true)
void Replayer::executeOp(const ThreadOp& op, bool useLock)
{
  const rcclApiCall& call = op.call;
  int opLineNum = op.lineNum;
  
  // Set correct GPU device for this operation
  HIP_CALL(hipSetDevice(call.hipDev));
  void *sbuffer = NULL, *rbuffer = NULL;
  
  // For collective ops, setup buffers and streams (with locking for thread safety)
  if (call.type < rrGroupStart) {
    if (useLock) resourceMtx.lock();  // Acquire lock before accessing shared maps
    
    // Allocate send buffer if not already allocated
    if (call.sendPtrBase && !dMemMap[call.sendPtrBase].base) {
      HIP_CALL(hipMalloc(&dMemMap[call.sendPtrBase].base, dMemMap[call.sendPtrBase].size));
    }
    // Calculate actual buffer address: base + offset from log
    if (call.sendPtrBase) {
      std::ptrdiff_t diff = (char*)call.sendbuff - (char*)call.sendPtrBase;
      sbuffer = (char*)dMemMap[call.sendPtrBase].base + diff;
    }
    
    // Allocate recv buffer if not already allocated
    if (call.recvPtrBase && !dMemMap[call.recvPtrBase].base) {
      HIP_CALL(hipMalloc(&dMemMap[call.recvPtrBase].base, dMemMap[call.recvPtrBase].size));
    }
    // Calculate actual buffer address: base + offset from log
    if (call.recvPtrBase) {
      std::ptrdiff_t diff = (char*)call.recvbuff - (char*)call.recvPtrBase;
      rbuffer = (char*)dMemMap[call.recvPtrBase].base + diff;
    }
    
    // Create stream if not exists, track which device it belongs to
    if (call.stream && !streams[call.stream].stream) {
      HIP_CALL(hipStreamCreate(&streams[call.stream].stream));
      streams[call.stream].device = call.hipDev;
    }
    
    if (useLock) resourceMtx.unlock();  // Release lock after setup
  }
  
  // Get actual stream (or nullptr for default stream)
  hipStream_t stream = call.stream ? streams[call.stream].stream : nullptr;
  
  // ========== GRAPH HANDLING: Before operation ==========
  // Check if this op is part of a graph (either capturing or launching)
  if (call.graphCaptured == 1) {
    // CAPTURING mode: ops are recorded into a graph, not executed immediately
    if (useLock) graphMtx.lock();
    graphLife[call.graphID].counter--;
    
    if (graphLife[call.graphID].starts.contains(opLineNum)) {
      // First op of graph: begin stream capture
      HIP_CALL(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
      graphLife[call.graphID].stream = call.stream;  // Track which stream owns the graph
      printf("[INFO    ] Rank %d Thread %d Line %d : Begin graph capture %llu\n", 
             myRank, call.tid, opLineNum, call.graphID);
    } else if (graphLife[call.graphID].stream != call.stream) {
      // Multi-stream graph: create event dependency to main capture stream
      printf("[WARNING ] Rank %d Thread %d Line %d : Multi-stream graph may not replay accurately\n",
             myRank, call.tid, opLineNum);
      hipEvent_t event;
      HIP_CALL(hipEventCreate(&event));
      graphLife[call.graphID].events.push_back(event);
      HIP_CALL(hipEventRecord(event, streams[graphLife[call.graphID].stream].stream));
      HIP_CALL(hipStreamWaitEvent(stream, event, 0));
    }
    if (useLock) graphMtx.unlock();
  } else if (call.graphID) {
    // LAUNCHING mode: graph was previously captured, now launching it
    if (useLock) graphMtx.lock();
    if (graphLife[call.graphID].starts.contains(opLineNum)) {
      // First op of graph launch: execute the graph
      printf("[INFO    ] Rank %d Thread %d Line %d : Launching graph %llu\n",
             myRank, call.tid, opLineNum, call.graphID);
      HIP_CALL(hipGraphLaunch(graphLife[call.graphID].graphExec, stream));
    }
    if (useLock) graphMtx.unlock();
    // Skip actual op execution - graph replays it
    printf("[INFO    ] Rank %d Thread %d Line %d : %s (played by graph %llu)\n",
           myRank, call.tid, opLineNum, rcclCallStr[call.type], call.graphID);
    return;
  }
  
  // Helper lambda to get comm with optional locking
  auto getComm = [&]() -> ncclComm_t {
    ncclComm_t comm;
    if (useLock) resourceMtx.lock();
    comm = commMap[call.comm];
    if (useLock) resourceMtx.unlock();
    return comm;
  };
  
  switch (call.type) {
  // NCCL collectives - executed sequentially in Phase 2a with MPI synchronization
  case rrAllReduce:
  {
    NCCL_CALL(ncclAllReduce(sbuffer, rbuffer, call.count, call.datatype, call.op, getComm(), stream));
    break;
  }
  
  case rrReduceScatter:
  {
    NCCL_CALL(ncclReduceScatter(sbuffer, rbuffer, call.count, call.datatype, call.op, getComm(), stream));
    break;
  }
  
  case rrReduce:
  {
    NCCL_CALL(ncclReduce(sbuffer, rbuffer, call.count, call.datatype, call.op, call.root, getComm(), stream));
    break;
  }
  
  case rrAllReduceWithBias:
  {
    std::vector<char> acc(call.count * ncclTypeSize(call.datatype));
    NCCL_CALL(ncclAllReduceWithBias(sbuffer, rbuffer, call.count, call.datatype, call.op, getComm(), stream, acc.data()));
    HIP_CALL(hipStreamSynchronize(stream));
    break;
  }
    
  case rrAllGather:
  {
    NCCL_CALL(ncclAllGather(sbuffer, rbuffer, call.count, call.datatype, getComm(), stream));
    break;
  }
  
  case rrAllToAll:
  {
    NCCL_CALL(ncclAllToAll(sbuffer, rbuffer, call.count, call.datatype, getComm(), stream));
    break;
  }
  
  case rrBroadcast:
  {
    NCCL_CALL(ncclBroadcast(sbuffer, rbuffer, call.count, call.datatype, call.root, getComm(), stream));
    break;
  }
  
  case rrBcast:
  {
    NCCL_CALL(ncclBcast(rbuffer, call.count, call.datatype, call.root, getComm(), stream));
    break;
  }
  
  case rrScatter:
  {
    NCCL_CALL(ncclScatter(sbuffer, rbuffer, call.count, call.datatype, call.root, getComm(), stream));
    break;
  }
  
  case rrGather:
  {
    NCCL_CALL(ncclGather(sbuffer, rbuffer, call.count, call.datatype, call.root, getComm(), stream));
    break;
  }
  
  case rrSend:
  {
    NCCL_CALL(ncclSend(rbuffer, call.count, call.datatype, call.root, getComm(), stream));
    break;
  }
  
  case rrRecv:
  {
    NCCL_CALL(ncclRecv(rbuffer, call.count, call.datatype, call.root, getComm(), stream));
    break;
  }
    
  // AllToAllv: variable-size AllToAll using pre-parsed count/displacement arrays
  case rrAllToAllv:
  {
    NCCL_CALL(ncclAllToAllv(sbuffer, op.sendcounts.data(), op.sdispls.data(),
                            rbuffer, op.recvcounts.data(), op.rdispls.data(),
                            call.datatype, getComm(), stream));
    break;
  }
    
  case rrMemAlloc:
  {
    if (useLock) resourceMtx.lock();
    NCCL_CALL(ncclMemAlloc(&dMemMap[call.recvbuff].base, call.count));
    if (useLock) resourceMtx.unlock();
    break;
  }
  
  case rrMemFree:
  {
    if (useLock) resourceMtx.lock();
    NCCL_CALL(ncclMemFree(dMemMap[call.recvbuff].base));
    if (useLock) resourceMtx.unlock();
    break;
  }
    
  case rrCommRegister:
  {
    if (useLock) resourceMtx.lock();
    if (!dMemMap[call.sendPtrBase].base) {
      HIP_CALL(hipMalloc(&dMemMap[call.sendPtrBase].base, dMemMap[call.sendPtrBase].size));
    }
    void* buf = (char*)dMemMap[call.sendPtrBase].base + 
                (std::ptrdiff_t)((char*)call.sendbuff - (char*)call.sendPtrBase);
    NCCL_CALL(ncclCommRegister(commMap[call.comm], buf, dMemMap[call.sendPtrBase].size, &handleMap[call.recvbuff]));
    if (useLock) resourceMtx.unlock();
    break;
  }
  
  case rrCommDeregister:
  {
    if (useLock) resourceMtx.lock();
    NCCL_CALL(ncclCommDeregister(commMap[call.comm], handleMap[call.recvbuff]));
    if (useLock) resourceMtx.unlock();
    break;
  }
  
  case rrCommFinalize:
  {
    if (useLock) resourceMtx.lock();
    NCCL_CALL(ncclCommFinalize(commMap[call.comm]));
    if (useLock) resourceMtx.unlock();
    break;
  }
  
  // CommSplit: create sub-communicator from existing communicator (collective!)
  case rrCommSplit:
  {
    int color = call.nRanks;
    int key = call.globalRank;
    ncclComm_t newcomm;
    ncclComm_t parentComm = (ncclComm_t) call.commId;
    if (useLock) resourceMtx.lock();
    NCCL_CALL(ncclCommSplit(commMap[parentComm], color, key, &newcomm, NULL));
    commMap[call.comm] = newcomm;
    if (useLock) resourceMtx.unlock();
    break;
  }
    
  case rrHipStreamSynchronize:
    if (stream) HIP_CALL(hipStreamSynchronize(stream));
    break;
  
  case rrHipDeviceSynchronize:
    HIP_CALL(hipDeviceSynchronize());
    break;
  
  case rrHipEventCreate:
  {
    hipEvent_t newEvent;
    HIP_CALL(hipEventCreate(&newEvent));
    if (useLock) resourceMtx.lock();
    eventMap[call.event] = newEvent;  // Map: logged_event_ptr -> new_event
    if (useLock) resourceMtx.unlock();
    break;
  }
  
  case rrHipEventDestroy:
  {
    hipEvent_t evt = nullptr;
    if (useLock) resourceMtx.lock();
    if (eventMap.contains(call.event)) {
      evt = eventMap[call.event];
      eventMap.erase(call.event);
    }
    if (useLock) resourceMtx.unlock();
    if (evt) HIP_CALL(hipEventDestroy(evt));
    break;
  }
  
  case rrHipEventRecord:
  {
    hipEvent_t evt = nullptr;
    if (useLock) resourceMtx.lock();
    if (eventMap.contains(call.event)) {
      evt = eventMap[call.event];
    }
    if (useLock) resourceMtx.unlock();
    if (evt && stream) {
      HIP_CALL(hipEventRecord(evt, stream));
    }
    break;
  }
  
  case rrHipStreamWaitEvent:
  {
    hipEvent_t evt = nullptr;
    if (useLock) resourceMtx.lock();
    if (eventMap.contains(call.event)) {
      evt = eventMap[call.event];
    }
    if (useLock) resourceMtx.unlock();
    if (evt && stream) {
      HIP_CALL(hipStreamWaitEvent(stream, evt, 0));
    }
    break;
  }
  
  case rrHipEventSynchronize:
  {
    hipEvent_t evt = nullptr;
    if (useLock) resourceMtx.lock();
    if (eventMap.contains(call.event)) {
      evt = eventMap[call.event];
    }
    if (useLock) resourceMtx.unlock();
    if (evt) {
      HIP_CALL(hipEventSynchronize(evt));
    }
    break;
  }
  
  case rrGroupStart:
    NCCL_CALL(ncclGroupStart());
    break;
  
  case rrGroupEnd:
    NCCL_CALL(ncclGroupEnd());
    break;
  
  // Not supported operations
  case rrGroupSimulatedEnd:
  case rrRedOpCreatePreMulSum:
  case rrRedOpDestroy:
  case rrOtherCall:
    printf("[WARN    ] Rank %d Thread %d Line %d : Unsupported op %s\n", 
           myRank, call.tid, opLineNum, rcclCallStr[call.type]);
    break;
  
  default:
    printf("[WARN    ] Rank %d Thread %d Line %d : Unknown op type %d\n", 
           myRank, call.tid, opLineNum, call.type);
  }
  
  // ========== GRAPH HANDLING: After operation ==========
  // Check if this was the last op in a graph capture
  if (call.graphCaptured == 1) {
    if (useLock) graphMtx.lock();
    
    // Handle multi-stream graph: sync back to main capture stream
    if (graphLife[call.graphID].stream != call.stream) {
      hipEvent_t event;
      HIP_CALL(hipEventCreate(&event));
      graphLife[call.graphID].events.push_back(event);
      HIP_CALL(hipEventRecord(event, stream));
      HIP_CALL(hipStreamWaitEvent(streams[graphLife[call.graphID].stream].stream, event, 0));
    }
    
    // Last op of graph: end capture and instantiate
    if (graphLife[call.graphID].counter == 0) {
      hipGraphNode_t errorNode;
      char errorLog[1024];
      hipStream_t captureStream = streams[graphLife[call.graphID].stream].stream;
      
      HIP_CALL(hipStreamEndCapture(captureStream, &graphLife[call.graphID].graph));
      HIP_CALL(hipGraphInstantiate(&graphLife[call.graphID].graphExec, 
                                   graphLife[call.graphID].graph, &errorNode, errorLog, 1024));
      
      // Cleanup temporary events used for multi-stream sync
      for (hipEvent_t e : graphLife[call.graphID].events) {
        HIP_CALL(hipEventDestroy(e));
      }
      graphLife[call.graphID].events.clear();
      
      printf("[INFO    ] Rank %d Thread %d Line %d : Graph %llu captured and instantiated\n",
             myRank, call.tid, opLineNum, call.graphID);
    }
    if (useLock) graphMtx.unlock();
  }
  
  printf("[INFO    ] Rank %d Thread %d Line %d : %s done\n", myRank, call.tid, opLineNum, rcclCallStr[call.type]);
}

int main(int argc, char **argv)
{
  unsetenv("RCCL_REPLAY_FILE");
  MPI_Init(&argc, &argv);
  if (argc <= 1) {
    printf("Usage: %s logfile [numGpusPerMpiRank = 1]\n", argv[0]);
    exit(1);
  }

  // Parse rank information
  int mpiRank, numMpiRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numMpiRanks);

  // Parse command line arguments
  char* logFilename       = argv[1];
  int   numGpusPerMpiRank = (argc > 2 ? atoi(argv[2]) : 1);
  /// int   parseOnly         = (argc > 3 ? atoi(argv[3]) : 0);
  assert(numGpusPerMpiRank == 1);

  // Figure out starting GPU index to use based on hostname
  int nameLen, pid;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(hostname, &nameLen);

  std::string output_file, output_extension;
  int json_format = ParseLogFormat(logFilename, output_file, output_extension);
  assert(json_format == 0);

  // Only root handles file-rank assignment to avoid file handle pressure
  if (mpiRank != 0)
  {
    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               NULL, 0, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);

    MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL,
                hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL,
                &pid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    std::vector<char> allhosts(numMpiRanks * MPI_MAX_PROCESSOR_NAME, 0);
    std::vector<int> pids(numMpiRanks * sizeof(int), 0);

    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               allhosts.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    // All hostnames in the recorded program
    std::unordered_set<std::string> hostnames;
    for (int i = 0; i < numMpiRanks; i++)
    {
      hostnames.insert(std::string(allhosts.data() + i * MPI_MAX_PROCESSOR_NAME)); // assuming null terminator included
    }

    // Register all hostnames and pid from recorder logs
    std::unordered_map<std::string, std::vector<int>> logHosts;
    int file_pid, a = 0/*counter*/;
    DIR *d;
    struct dirent *dir;
    if (d = opendir(".")) {
      while ((dir = readdir(d)) != NULL) {
        // MPI_MAX_PROCESSOR_NAME = 256
        if (sscanf(dir->d_name, (output_file + ".%d.%256[^.]" + output_extension).c_str(), &file_pid, hostname) == 2)
        {
          logHosts[std::string(hostname)].push_back(file_pid);
          a++;
        }
      }
      closedir(d);
    }
    // Double check number of nodes and number of processes match for recorder and replayer
    assert(logHosts.size() == hostnames.size());
    assert(a == numMpiRanks);
    // Assign mapping of replayer hostname to recorder hostname
    std::unordered_map<std::string, std::string> hostAssignment;
    auto it = logHosts.begin();
    for (const auto &host : hostnames)
    {
      hostAssignment[host] = (*it).first;
      it++;
    }
    for (int i = 0; i < numMpiRanks; i++)
    {
      std::string host(allhosts.data() + i * MPI_MAX_PROCESSOR_NAME);
      strcpy(allhosts.data() + i * MPI_MAX_PROCESSOR_NAME, hostAssignment[host].c_str());
      pids[i] = logHosts[hostAssignment[host]].back();
      logHosts[hostAssignment[host]].pop_back();
    }

    // Distribute the target log for each rank (pid and hostname)
    MPI_Scatter(allhosts.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(pids.data(), 1, MPI_INT,
                &pid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  // Initialize Replayer
  std::string logfile = output_file + "." + std::to_string(pid) + "." +
                        std::string(hostname) + output_extension; /// perhaps another func for assemble logname
  std::cout << mpiRank << " : " << logfile<<std::endl;
  Replayer replayer(logfile, json_format, mpiRank, numMpiRanks);

  if (mpiRank == 0)
    printf("RCCL Replayer version 0: %d ranks x %d gpu/Rank\n", numMpiRanks, numGpusPerMpiRank);
  printf("Rank %d [%s]\n", mpiRank, hostname);

  replayer.parse();
  printf("Rank %d parsing completed, starting replay\n", mpiRank);
  replayer.replay();
  MPI_Finalize();
  return 0;
}
