#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <mpi.h>
#include <fstream>

#include "rcclReplayer.hpp"

int main(int argc, char **argv)
{
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
  int   parseOnly         = (argc > 3 ? atoi(argv[3]) : 0);

  CollectiveCalls collCalls;
  collCalls.firstGlobalRank = mpiRank * numGpusPerMpiRank;
  collCalls.numGlobalRanks  = numMpiRanks * numGpusPerMpiRank;

  // Figure out starting GPU index to use based on hostname
  int nameLen;
  char name[MPI_MAX_PROCESSOR_NAME];
  std::vector<char> allnames(numMpiRanks * MPI_MAX_PROCESSOR_NAME, 0);
  MPI_Get_processor_name(name, &nameLen);
  MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                allnames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);

  // Offset local gpu device index based on number of previous ranks on the same host
  collCalls.localGpuOffset  = 0;
  for (int rank = 0; rank < mpiRank; rank++) {
    if (!strcmp(name, allnames.data() + (rank * MPI_MAX_PROCESSOR_NAME)))
      collCalls.localGpuOffset += numGpusPerMpiRank;
  }
  if (mpiRank == 0)
    printf("RCCL Replayer: %d x %d = %d total ranks\n", numMpiRanks, numGpusPerMpiRank, collCalls.numGlobalRanks);
  printf("Rank %d [%s] LocalGpuOffset: %d GlobalRankFirst %d GlobalRankLast %d\n",
         mpiRank, name, collCalls.localGpuOffset, collCalls.firstGlobalRank, collCalls.firstGlobalRank + numGpusPerMpiRank - 1);

  // Parse collectives from logfile
  if (parseOnly) collCalls.numGlobalRanks = parseOnly;
  ParseCollectives(logFilename, mpiRank == 0, collCalls);
  if (collCalls.groupCalls.size() == 0) {
    MPI_Finalize();
    return 0;
  }
  if (parseOnly) return 0;

  // Setup all communicators
  if (mpiRank == 0) printf("Preparing %d communicator(s) per rank\n", collCalls.numCommsPerRank);
  collCalls.localRankComms.resize(numGpusPerMpiRank, std::vector<ncclComm_t>(collCalls.numCommsPerRank));
  collCalls.localRankStreams.resize(numGpusPerMpiRank, std::vector<hipStream_t>(collCalls.numCommsPerRank));

  for (int commIdx = 0; commIdx < collCalls.numCommsPerRank; commIdx++) {
    // Create a unique ID and broadcast it to all ranks
    ncclUniqueId uniqueId;
    if (mpiRank == 0) ncclGetUniqueId(&uniqueId);
    MPI_Bcast(&uniqueId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize comms and strams
    NCCL_CALL(ncclGroupStart());
    for (int i = 0; i < numGpusPerMpiRank; i++) {
      HIP_CALL(hipSetDevice(collCalls.localGpuOffset + i));
      NCCL_CALL(ncclCommInitRank(&collCalls.localRankComms[i][commIdx], collCalls.numGlobalRanks, uniqueId, collCalls.firstGlobalRank + i));
      HIP_CALL(hipStreamCreate(&collCalls.localRankStreams[i][commIdx]));
    }
    NCCL_CALL(ncclGroupEnd());
  }
  printf("Rank %d Done setting up communicators\n", mpiRank);

  int numSkippedCalls = 0;
  double runTime;
  std::ofstream datafile;
  datafile.open("replayer_data.csv");
  if (!datafile.is_open()) {
    printf("[ERROR] Unable to open file replayer_data.csv\n");
    exit(-1);
  }
  datafile << "callNumber, functionName, inPlace, count(numElements), datatype, op, root, time(msec), groupCallBusBandwidth(GB/s)\n";
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < collCalls.groupCalls.size(); i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (collCalls.groupCalls[i].isValid) {
      if (mpiRank == 0)
      {
        printf("Running Collective Call %lu of %lu\n", i+1, collCalls.groupCalls.size());
        PrintGroupCall(collCalls.groupCalls[i]);
      }
      double runTime = ReplayRccl(collCalls, i);
      if (mpiRank == 0) {
        dataToCsv(collCalls.groupCalls[i], datafile, runTime);
      }
    } else {
      if (mpiRank == 0) {
        printf("[ERROR] in group call: (skipping...)\n");
        for (auto const& rd : collCalls.groupCalls[i].rankData) {
          printf("  - Rank %02d: comm %d in line %d\n", rd.first, rd.second.commIdx, rd.second.lineNum);
          for (int task = 0; task < rd.second.tasks.size(); task++) {
            TaskInfo ti = rd.second.tasks[task];
            printf("  - Task %02d: %32s inPlace=%d count=%lu datatype=%d op=%d root=%d\n",
                   task, ncclFuncNames[ti.funcType], ti.inPlace, ti.count, ti.datatype, ti.op, ti.root);
          }
        }
      }
      numSkippedCalls++;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  datafile.close();

  // Destroy all communicators
  for (int commIdx = 0; commIdx < collCalls.numCommsPerRank; commIdx++) {
    for (int i = 0; i < numGpusPerMpiRank; i++) {
      NCCL_CALL(ncclCommDestroy(collCalls.localRankComms[i][commIdx]));
      HIP_CALL(hipStreamDestroy(collCalls.localRankStreams[i][commIdx]));
    }
  }

  if (mpiRank == 0) printf("Executed group calls: %zu\n", collCalls.groupCalls.size() - numSkippedCalls);
  if (mpiRank == 0) printf("Skipped group calls: %d\n", numSkippedCalls);

  // Time it takes to execute all the group calls
  if (mpiRank == 0) printf("Execution Time: %f seconds\n", duration.count());
  printf("MPI Rank %d Success\n", mpiRank);

  MPI_Finalize();
  return 0;
}

void PrintGroupCall(GroupCall const& gc)
{
  printf("OpCount: %d\n", gc.opCount);

  for (auto rd : gc.rankData) {
    printf("  - Rank %02d: comm %d\n", rd.first, rd.second.commIdx);

    for (int task = 0; task < rd.second.tasks.size(); task++) {
      TaskInfo ti = rd.second.tasks[task];
      std::string funcName = (ti.funcType == ncclCollSend || ti.funcType == ncclCollRecv) ? "Send/Recv" : ncclFuncNames[ti.funcType];
      printf("  - Task %02d: %32s inPlace=%d count=%lu datatype=%d op=%d root=%d\n",
             task, funcName.c_str(), ti.inPlace, ti.count, ti.datatype, ti.op, ti.root);
    }
  }
}


void dataToCsv(GroupCall const& gc, std::ofstream &datafile, double runTime)
{
  auto rd = *(gc.rankData.begin());
  TaskInfo ti = rd.second.tasks[0];
  std::string funcName = (ti.funcType == ncclCollSend || ti.funcType == ncclCollRecv) ? "Send/Recv" : ncclFuncNames[ti.funcType];
  double n = (double) (ti.count);
  double S = (double) (n * (double)DataTypeToBytes(ti.datatype));
  double t = (double) (runTime/1000); //milliseconds to seconds
  double busBw = (S/t);
  if (funcName == "AllReduce") busBw *= (2*(n- 1)/n);
  else if (funcName == "ReduceScatter" || funcName == "AllGather") busBw *= ((n-1)/n);
  busBw /= (1e9); //in gb/s
  std::string dataTypeName = DataTypeToName(ti.datatype);
  std::string redOp = getRedOp(ti.op);
  datafile << gc.opCount << ", " << funcName.c_str() << ", " << ti.inPlace << ", " << ti.count << ", " << dataTypeName << ", " << redOp << ", " << ti.root << ", " << runTime << ", " << busBw << "\n";
}

void ParseCollectives(char const* logFilename, bool isFirstRank, CollectiveCalls& cc)
{
  bool verbose = isFirstRank && (getenv("VERBOSE") != NULL);
  cc.globalRankComms.clear();
  cc.globalRankComms.resize(cc.numGlobalRanks);
  cc.groupCalls.clear();

  FILE* fp = fopen(logFilename, "r");
  if (!fp) {
    printf("[ERROR] Unable to open file %s\n", logFilename);
    exit(-1);
  }

  char line[2048];
  LineItem li;
  int lineNum = 0;

  while (fgets(line, 2048, fp)) {
    ++lineNum;

    //Ignore invalid lines and collectives
    if (!ParseLineItem(line, li) || li.nRanks != cc.numGlobalRanks) continue;

    // Figure out commIdx for this globalrank
    int commIdx = -1;
    for (auto i = 0; i < cc.globalRankComms[li.globalRank].size(); i++) {
      if (!strcmp(cc.globalRankComms[li.globalRank][i].c_str(), li.comm)) {
        commIdx = i;
        break;
      }
    }
    if (commIdx == -1) {
      commIdx = cc.globalRankComms[li.globalRank].size();
      cc.globalRankComms[li.globalRank].push_back(li.comm);
    }

    TaskInfo taskInfo;
    taskInfo.funcType = GetFuncType(li.opName);
    taskInfo.inPlace  = !strcmp(li.sendbuff, li.recvbuff);
    taskInfo.count    = li.count;
    taskInfo.datatype = (ncclDataType_t) li.datatype;
    taskInfo.op       = (ncclRedOp_t) li.op;
    taskInfo.root     = li.root;

    // Find the appropriate GroupCall that this task belongs to
    // If it doesn't exist yet, then create it
    bool found = false;
    for (auto& gc : cc.groupCalls) {
      if (gc.opCount != li.opCount) continue;
      if (gc.rankData.count(li.globalRank)) {
        RankData& rd = gc.rankData[li.globalRank];
        if (rd.commIdx != commIdx || rd.tasks.size() != li.task)
          continue;

        rd.tasks.push_back(taskInfo);
        found = true;
        break;
      }
      // Rank has no tasks - make sure this is task 0
      else if (li.task == 0) {
        gc.rankData[li.globalRank].lineNum = lineNum;
        gc.rankData[li.globalRank].commIdx = commIdx;
        gc.rankData[li.globalRank].tasks.push_back(taskInfo);
        found = true;
        break;
      }
    }

    // If no collectives were found, create new one
    if (!found) {
      if (li.task != 0) {
        if (isFirstRank) printf("[WARN] Was unable to find corresponding collective for line %d\n", lineNum);
      }

      GroupCall gc;
      gc.opCount = li.opCount;
      gc.rankData[li.globalRank].commIdx = commIdx;
      gc.rankData[li.globalRank].lineNum = lineNum;
      gc.rankData[li.globalRank].tasks.push_back(taskInfo);
      cc.groupCalls.push_back(gc);
    }
  }
  fclose(fp);

  // Validate group calls
  // - For non Send/Recv, check that all ranks participate with same parameters count
  // - For Send/Recv, check that pairs of Send/Recv calls exist
  if (isFirstRank) printf("Found %lu groupCalls\n", cc.groupCalls.size());
  for (int i = 0; i < cc.groupCalls.size(); i++) {
    GroupCall& gc = cc.groupCalls[i];
    std::map<std::tuple<std::string, size_t, int, int>, std::vector<int>> arrivalCounter;

    gc.isValid = true;

    for (auto rd : gc.rankData) {
      for (int task = 0; task < rd.second.tasks.size(); task++) {
        TaskInfo ti = rd.second.tasks[task];

        std::string funcName = (ti.funcType == ncclCollSend || ti.funcType == ncclCollRecv) ? "Send/Recv" : ncclFuncNames[ti.funcType];
        std::tuple<std::string, size_t, int, int> key(funcName, ti.count, ti.datatype, ti.op);

        auto& rankVector = arrivalCounter[key];
        if (rankVector.size() < cc.numGlobalRanks)
          rankVector.resize(cc.numGlobalRanks);

        // rankVector<int> in arrivalCount represents the rank information
        // Count the number of tasks that are going to be executed by each rank. This is to validate the group call later on.
        // Nom-Send/Recv rank counts (rankVector<int> elements) should be equal at the end, and for Send/Recv, all the elements of rankVector<int> should be equal to 0
        if (ti.funcType == ncclCollRecv) {
          rankVector[ti.root]--;
        } else {
          rankVector[rd.first]++;
        }
      }
    }

    // Iterate through the map variable and report/validate the results
    for (const auto& e : arrivalCounter) {
      int maxVal;
      std::string funcName = std::get<0>(e.first);
      size_t      count    = std::get<1>(e.first);
      int const   datatype = std::get<2>(e.first);
      int const   op       = std::get<3>(e.first);

      bool isp2p = (funcName == "Send/Recv");
      if (!isp2p) maxVal = *std::max_element(e.second.begin(), e.second.end());

      // Validate all the ranks have required amount of collective call (task)
      for (int i = 0; i < e.second.size(); i++) {
        if (e.second[i] != (isp2p ? 0 : maxVal)) {
          std::string warning = (isp2p ? (e.second[i] > 0 ? "[WARN] Missing Recv" : "[WARN] Missing Send") : "[WARN] Missing " + std::string(funcName))
            + " count=" + std::to_string(count) + " datatype=" + std::to_string(datatype) + " op=" + std::to_string(op) + " at rank [" + std::to_string(i) + "]";
          if(isFirstRank) printf("%s\n", warning.c_str());

          gc.isValid = false;
        }
      }
    }
  }

  // Check number of comms per rank
  cc.numCommsPerRank = cc.globalRankComms[0].size();
  for (int i = 1; i < cc.numGlobalRanks; i++) {
    if (cc.numCommsPerRank != cc.globalRankComms[i].size()) {
      printf("[ERROR] Replayer currently only supports identical number of communicators across all ranks\n");
      printf("[ERROR] Rank %d has %lu communicators (expecting %d)\n", i, cc.globalRankComms[i].size(), cc.numCommsPerRank);
      exit(1);
    }
  }
}

bool ParseLineItem(char const* line, LineItem& li)
{
  return sscanf(line,
                "%[^:]:%d:%d [%d] NCCL INFO %[^:]: opCount %x sendbuff %s "
                "recvbuff %s count %lu datatype %d op %d root %d comm %s "
                "[nranks=%d] stream %p task %d globalrank %d",
                li.hostname, &li.pid, &li.tid, &li.cudaDev, li.opName,
                &li.opCount, li.sendbuff, li.recvbuff,
                &li.count, &li.datatype, &li.op, &li.root, li.comm,
                &li.nRanks, &li.stream, &li.task, &li.globalRank) == 17;
}

double ReplayRccl(CollectiveCalls const& cc, int groupIdx)
{
  int numLocalRanks = cc.localRankComms.size();

  // Allocate memory for collective
  std::vector<std::vector<void*>> sendbuff(numLocalRanks);
  std::vector<std::vector<void*>> recvbuff(numLocalRanks);

  for (int localIdx = 0; localIdx < numLocalRanks; localIdx++) {
    int globalRank = cc.firstGlobalRank + localIdx;
    if (cc.groupCalls[groupIdx].rankData.count(globalRank) == 0) continue;
    HIP_CALL(hipSetDevice(cc.localGpuOffset + localIdx));

    RankData const& rankData = cc.groupCalls[groupIdx].rankData.at(globalRank);
    int numTasks = rankData.tasks.size();
    sendbuff[localIdx].resize(numTasks);
    recvbuff[localIdx].resize(numTasks);

    for (int taskId = 0; taskId < numTasks; taskId++) {
      TaskInfo const& task = rankData.tasks[taskId];

      // Each task has a size based on the type of collective (funcType)
      std::pair<size_t, size_t> numBytes = GetSize(task, cc.numGlobalRanks);

      if (task.inPlace) {
        numBytes.first = std::max(numBytes.first, numBytes.second);
        numBytes.second = numBytes.first;
      }

      // Set the device and allocate send/recv buffers
      HIP_CALL(hipMalloc(&sendbuff[localIdx][taskId], numBytes.first));
      HIP_CALL(hipMemset(sendbuff[localIdx][taskId], 0, numBytes.first));

      if (!task.inPlace) {
        HIP_CALL(hipMalloc(&recvbuff[localIdx][taskId], numBytes.second));
        HIP_CALL(hipMemset(recvbuff[localIdx][taskId], 0, numBytes.second));
      } else {
        recvbuff[localIdx][taskId] = sendbuff[localIdx][taskId];
      }

      HIP_CALL(hipDeviceSynchronize());
    }
  }

  // Execute the collective call (task)
  std::chrono::time_point start = std::chrono::high_resolution_clock::now();
  NCCL_CALL(ncclGroupStart());
  for (int localIdx = 0; localIdx < numLocalRanks; localIdx++) {
    int globalRank = cc.firstGlobalRank + localIdx;
    if (cc.groupCalls[groupIdx].rankData.count(globalRank) == 0) continue;

    RankData const& rankData = cc.groupCalls[groupIdx].rankData.at(globalRank);
    int numTasks = rankData.tasks.size();
    int commIdx = rankData.commIdx;
    for (int taskId = 0; taskId < numTasks; taskId++) {
      TaskInfo const& task = rankData.tasks[taskId];
      ExecuteCollective(task, cc.localRankComms[localIdx][commIdx], cc.localRankStreams[localIdx][commIdx],
                        sendbuff[localIdx][taskId],
                        recvbuff[localIdx][taskId]);
    }
  }
  NCCL_CALL(ncclGroupEnd());

  // Synchronize devices and free memory
  for (int localIdx = 0; localIdx < numLocalRanks; localIdx++) {
    int globalRank = cc.firstGlobalRank + localIdx;
    if (cc.groupCalls[groupIdx].rankData.count(globalRank) == 0) continue;

    RankData const& rankData = cc.groupCalls[groupIdx].rankData.at(globalRank);
    int commIdx = rankData.commIdx;
    HIP_CALL(hipStreamSynchronize(cc.localRankStreams[localIdx][commIdx]));
  }

  std::chrono::time_point end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = (end - start);
  double runTime = duration.count();
  runTime *= 1000; //convering into milliseconds

  for (int localIdx = 0; localIdx < numLocalRanks; localIdx++) {
    int globalRank = cc.firstGlobalRank + localIdx;
    RankData const& rankData = cc.groupCalls[groupIdx].rankData.at(globalRank);
    int numTasks = rankData.tasks.size();
    for (int taskId = 0; taskId < numTasks; taskId++) {
      TaskInfo const& task = rankData.tasks[taskId];
      HIP_CALL(hipFree(sendbuff[localIdx][taskId]));
      if (!task.inPlace) HIP_CALL(hipFree(recvbuff[localIdx][taskId]));
    }
  }
  return runTime;
}

// GetSize will return a pair of bytes where first element in pair represents bytesSent and the second bytesRecv
std::pair<size_t, size_t> GetSize(TaskInfo taskInfo, int numGlobalRanks) {
  size_t sendNumBytes, recvNumBytes;

  switch (taskInfo.funcType) {
  case ncclCollBroadcast: case ncclCollReduce: case ncclCollAllReduce:
    sendNumBytes = taskInfo.count * DataTypeToBytes(taskInfo.datatype);
    recvNumBytes = sendNumBytes;
    break;
  case ncclCollAllGather: case ncclCollGather:
    sendNumBytes = taskInfo.count * DataTypeToBytes(taskInfo.datatype);
    recvNumBytes = numGlobalRanks * sendNumBytes;
    break;
  case ncclCollReduceScatter: case ncclCollScatter:
    recvNumBytes = taskInfo.count * DataTypeToBytes(taskInfo.datatype);
    sendNumBytes = numGlobalRanks * recvNumBytes;
    break;
  case ncclCollAllToAll:
    sendNumBytes = numGlobalRanks * taskInfo.count * DataTypeToBytes(taskInfo.datatype);
    recvNumBytes = sendNumBytes;
    break;
  default:
    sendNumBytes = taskInfo.count * DataTypeToBytes(taskInfo.datatype);
    recvNumBytes = sendNumBytes;
  }
  return std::make_pair(sendNumBytes, recvNumBytes);
}

void ExecuteCollective(TaskInfo const& task, ncclComm_t const& comm, hipStream_t stream, const void *sendbuff, void *recvbuff)
{
  switch (task.funcType) {
  case ncclCollAllGather:
    NCCL_CALL(ncclAllGather(sendbuff, recvbuff, task.count, task.datatype, comm, stream));
    break;
  case ncclCollAllReduce:
    NCCL_CALL(ncclAllReduce(sendbuff, recvbuff, task.count, task.datatype, task.op, comm, stream));
    break;
  case ncclCollBroadcast:
    NCCL_CALL(ncclBroadcast(sendbuff, recvbuff, task.count, task.datatype, task.root, comm, stream));
    break;
  case ncclCollReduce:
    NCCL_CALL(ncclReduce(sendbuff, recvbuff, task.count, task.datatype, task.op, task.root, comm, stream));
    break;
  case ncclCollReduceScatter:
    NCCL_CALL(ncclReduceScatter(sendbuff, recvbuff, task.count, task.datatype, task.op, comm, stream));
    break;
  case ncclCollGather:
    NCCL_CALL(ncclGather(sendbuff, recvbuff, task.count, task.datatype, task.root, comm, stream));
    break;
  case ncclCollScatter:
    NCCL_CALL(ncclScatter(sendbuff, recvbuff, task.count, task.datatype, task.root, comm, stream));
    break;
  case ncclCollAllToAll:
    NCCL_CALL(ncclAllToAll(sendbuff, recvbuff, task.count, task.datatype, comm, stream));
    break;
  case ncclCollSend:
    NCCL_CALL(ncclSend(sendbuff, task.count, task.datatype, task.root, comm, stream));
    break;
  case ncclCollRecv:
    NCCL_CALL(ncclRecv(recvbuff, task.count, task.datatype, task.root, comm, stream));
    break;
  default:
    printf("Error: unsupported collective\n");
    exit(1);
  }
}
