#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <mpi.h>

#include "rcclReplayer.hpp"

bool ParseLineItem(char const* line, LineItem& li)
{
    return sscanf(line,
                    "%[^:]:%d:%d [%d] NCCL INFO %[^:]: opCount %d sendbuff %s "
                    "recvbuff %s count %lu datatype %d op %d root %d comm %s "
                    "[nranks=%d] stream %p task %d globalrank %d",
                    li.hostname, &li.pid, &li.tid, &li.cudaDev, li.opName,
                    &li.opCount, li.sendbuff, li.recvbuff,
                    &li.count, &li.datatype, &li.op, &li.root, li.comm,
                    &li.nRanks, &li.stream, &li.task, &li.globalRank) == 17;
}

void ParseCollectives(char const* logFilename, int const numGlobalRanks, std::vector<GroupCall>& groupCalls) {
    int mpiRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    groupCalls.clear();

    FILE *fp = fopen(logFilename, "r");
    if (!fp) {
        printf("[ERROR] Unable to open file %s\n", logFilename);
        exit(-1);
    }

    char line[1000];
    LineItem li;
    int lineNum = 0;
    while (fgets(line, 1000, fp)) {
        ++lineNum;

        //Ignore invalid lines and collectives
        if (!ParseLineItem(line, li) || li.nRanks != numGlobalRanks) continue;

        TaskInfo taskInfo;
        taskInfo.funcType   = GetFuncType(li.opName);
        taskInfo.inPlace    = !strcmp(li.sendbuff, li.recvbuff);
        taskInfo.count      = li.count;
        taskInfo.datatype   = (ncclDataType_t) li.datatype;
        taskInfo.op         = (ncclRedOp_t) li.op;
        taskInfo.root       = li.root;

        // Find the appropriate GroupCall that this task belongs to
        // If it doesn't exist yet, then create it
        bool found = false;
        for (auto& gc : groupCalls) {
            if (gc.rankData.count(li.globalRank)) {
                RankData& rd = gc.rankData[li.globalRank];
                if (rd.comm != li.comm || rd.tasks.size() != li.task)
                    continue;
                
                rd.tasks.push_back(taskInfo);
                found = true;
                break;
            }
            // Rank has no tasks - make sure this is task 0
            else if (li.task == 0) {
                gc.rankData[li.globalRank].comm = li.comm;
                gc.rankData[li.globalRank].lineNum = lineNum;
                gc.rankData[li.globalRank].tasks.push_back(taskInfo);
                found = true;
                break;
            }
        }

        // If no collectives were found, create new one
        if (!found) {
            if (li.task != 0) {
                if (mpiRank == 0) printf("[WARN] Was unable to find corresponding collective for line %d\n", lineNum);
            }

            groupCalls.resize(groupCalls.size() + 1);
            GroupCall& gc = groupCalls.back();
            gc.opCount = li.opCount;
            gc.rankData[li.globalRank].comm = li.comm;
            gc.rankData[li.globalRank].lineNum = lineNum;
            gc.rankData[li.globalRank].tasks.push_back(taskInfo);
        }
    }

    // - For non Send/Recv, check that all ranks participate with same parameters count
    // - For Send/Recv, check that pairs of Send/Recv calls exist
    if (mpiRank == 0) printf("Found %lu groupCalls\n", groupCalls.size());
    for (int i = 0; i < groupCalls.size(); i++) {
        GroupCall& gc = groupCalls[i];
        std::map<std::tuple<const char*, size_t, int, int>, std::vector<int>> arrivalCounter;

        gc.isValid = true;

        if (mpiRank == 0) {
            printf("GroupCall %d\n", i);
            printf(" - OpCount: %d\n", gc.opCount);
        }

        for (auto rd : gc.rankData) {
            if (mpiRank == 0) {
                printf("  - Rank %02d: comm %s\n", rd.first, rd.second.comm.c_str());
            }

            for (int task = 0; task < rd.second.tasks.size(); task++) {
                TaskInfo ti = rd.second.tasks[task];
                const char* funcName;

                if (ti.funcType == ncclCollSend || ti.funcType == ncclCollRecv)
                    funcName = "Send/Recv";
                else 
                    funcName = ncclFuncNames[ti.funcType];

                std::tuple<const char*, size_t, int, int> key(funcName, ti.count, ti.datatype, ti.op);

                if (mpiRank == 0) {
                    printf("  - Task %02d: %32s inPlace=%d count=%lu datatype=%d op=%d root=%d\n",
                        task, funcName, ti.inPlace, ti.count, ti.datatype, ti.op, ti.root);
                }

                auto& rankVector = arrivalCounter[key];
                if (rankVector.size() < numGlobalRanks) {
                    rankVector.resize(numGlobalRanks);
                }

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
            const char* funcName = std::get<0>(e.first);
            size_t count = std::get<1>(e.first);
            int datatype = std::get<2>(e.first);
            int op = std::get<3>(e.first);
            
            bool isp2p = (strcmp(std::get<0>(e.first), "Send/Recv") == 0);
            if (!isp2p) maxVal = *std::max_element(e.second.begin(), e.second.end());
            
            // Validate all the ranks have required amount of collective call (task)
            for (int i = 0; i < e.second.size(); i++) {
                if (e.second[i] != (isp2p ? 0 : maxVal)) {
                    std::string warning = (isp2p ? (e.second[i] > 0 ? "[WARN] Missing Recv" : "[WARN] Missing Send") : "[WARN] Missing " + std::string(funcName)) 
                            + " count=" + std::to_string(count) + " datatype=" + std::to_string(datatype) + " op=" + std::to_string(op) + " at rank [" + std::to_string(i) + "]";
                    if(mpiRank == 0) printf("%s\n", warning.c_str());

                    gc.isValid = false;
                }
            }
        }
    }
}

// GetSize will return a pair of bytes where first element in pair represents bytesSent and the second bytesRecv
std::pair<size_t, size_t> GetSize(TaskInfo taskInfo, int numGlobalRanks) {
    size_t sendNumBytes;
    size_t recvNumBytes;

    if (taskInfo.funcType == ncclCollBroadcast || taskInfo.funcType == ncclCollReduce || taskInfo.funcType == ncclCollAllReduce) {
        sendNumBytes = taskInfo.count * DataTypeToBytes(taskInfo.datatype);
        recvNumBytes = sendNumBytes;
    } else if (taskInfo.funcType == ncclCollAllGather || taskInfo.funcType == ncclCollGather) {
        sendNumBytes = taskInfo.count * DataTypeToBytes(taskInfo.datatype);
        recvNumBytes = numGlobalRanks * sendNumBytes;
    } else if (taskInfo.funcType == ncclCollReduceScatter || taskInfo.funcType == ncclCollScatter) {
        recvNumBytes = taskInfo.count * DataTypeToBytes(taskInfo.datatype); 
        sendNumBytes = numGlobalRanks * recvNumBytes; 
    } else if (taskInfo.funcType == ncclCollAllToAll) {
        sendNumBytes = numGlobalRanks * taskInfo.count * DataTypeToBytes(taskInfo.datatype);
        recvNumBytes = sendNumBytes;
    } else {
        sendNumBytes = taskInfo.count * DataTypeToBytes(taskInfo.datatype);
        recvNumBytes = sendNumBytes;
    }
    return std::make_pair(sendNumBytes, recvNumBytes);
}

void ExecuteCollective(TaskInfo task, ncclComm_t comm, hipStream_t stream, const void *sendbuff, void *recvbuff) {

    int funcTypeValue = (int)task.funcType;

    switch (funcTypeValue) {
        case ncclCollAllGather:
            NCCLCHECK(ncclAllGather(sendbuff, recvbuff, task.count, task.datatype, comm, stream));
            break;
        case ncclCollAllReduce:
            NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, task.count, task.datatype, task.op, comm, stream));
            break;
        case ncclCollBroadcast:
            NCCLCHECK(ncclBroadcast(sendbuff, recvbuff, task.count, task.datatype, task.root, comm, stream));
            break;
        case ncclCollReduce:
            NCCLCHECK(ncclReduce(sendbuff, recvbuff, task.count, task.datatype, task.op, task.root, comm, stream));
            break;
        case ncclCollReduceScatter:
            NCCLCHECK(ncclReduceScatter(sendbuff, recvbuff, task.count, task.datatype, task.op, comm, stream));
            break;
        case ncclCollGather:
            NCCLCHECK(ncclGather(sendbuff, recvbuff, task.count, task.datatype, task.root, comm, stream));
            break;
        case ncclCollScatter:
            NCCLCHECK(ncclScatter(sendbuff, recvbuff, task.count, task.datatype, task.root, comm, stream));
            break;
        case ncclCollAllToAll:
            NCCLCHECK(ncclAllToAll(sendbuff, recvbuff, task.count, task.datatype, comm, stream));
            break;
        case ncclCollSend:
            NCCLCHECK(ncclSend(sendbuff, task.count, task.datatype, task.root, comm, stream));
            break;
        case ncclCollRecv:
            NCCLCHECK(ncclRecv(recvbuff, task.count, task.datatype, task.root, comm, stream));
            break;
        default:
            printf("Error: unsupported collective\n");
            exit(1);
    }
}

void ReplayRccl(GroupCall& groupCall, std::vector<ncclComm_t> comms, std::vector<hipStream_t> streams,
                                                                     int const localGpuOffset, int const numGpusPerMpiRank, int const firstGlobalRank, int const numGlobalRanks) {
    
    std::vector<std::vector<void*>> sendbuff(numGpusPerMpiRank);
    std::vector<std::vector<void*>> recvbuff(numGpusPerMpiRank);

    NCCLCHECK(ncclGroupStart());
    for (int localIdx = 0; localIdx < numGpusPerMpiRank; localIdx++) {
        int globalRank = firstGlobalRank + localIdx;
        RankData& rankData = groupCall.rankData[globalRank];
    
        for (auto task : rankData.tasks) {
            void* sendBuffer;
            void* recvBuffer;

            // Each task has a size based on the type of collective (funcType)
            std::pair<size_t, size_t> numBytes = GetSize(task, numGlobalRanks);

            if (task.inPlace) {
                numBytes.first = std::max(numBytes.first, numBytes.second);
                numBytes.second = numBytes.first;
            }
            
            // Set the device and allocate send/recv buffers
            HIPCALL(hipSetDevice(localGpuOffset + localIdx));
            HIPCALL(hipMalloc(&sendBuffer, numBytes.first));
            HIPCALL(hipMalloc(&recvBuffer, numBytes.second));
            HIPCALL(hipMemset(sendBuffer, 0, numBytes.first));
            HIPCALL(hipMemset(recvBuffer, 0, numBytes.second));
            HIPCALL(hipDeviceSynchronize());

            // Add the send and receive buffers to their respective vectors
            sendbuff[localIdx].push_back(sendBuffer);
            recvbuff[localIdx].push_back(recvBuffer);

            // Execute the collective call (task)
            ExecuteCollective(task, comms[localIdx], streams[localIdx], sendBuffer, recvBuffer);
        }
    }
    NCCLCHECK(ncclGroupEnd());
    
    // Synchronize devices
    for (int i = 0; i < numGpusPerMpiRank; i++) {
        HIPCALL(hipStreamSynchronize(streams[i]));
    }

    // Free device memory for each task on each GPU
    for (int i = 0; i < numGpusPerMpiRank; i++) {
        for (auto& sendBuffer : sendbuff[i]) HIPCALL(hipFree(sendBuffer));
        for (auto& recvBuffer : recvbuff[i]) HIPCALL(hipFree(recvBuffer));
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    if (argc <= 1) {
        printf("Usage: %s logfile [numGpusPerMpiRank = 1]\n", argv[0]);
        exit(1);
    }

    // Parse rank information
    int mpiRank, numMpiRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numMpiRanks);
    
    // Default value for numGpusPerMpiRank is 1
    char* logFilename       = argv[1];
    int   numGpusPerMpiRank = (argc > 2 ? atoi(argv[2]) : 1);
    int   numGlobalRanks    = numMpiRanks * numGpusPerMpiRank;

    if (mpiRank == 0)
        printf("RCCL Replayer: %d x %d = %d total ranks\n", numMpiRanks, numGpusPerMpiRank, numGlobalRanks);
    
    // Parse logfile for Collectives
    std::vector<GroupCall> groupCalls;
    ParseCollectives(logFilename, numGlobalRanks, groupCalls);

    int localGpuOffset = 0;
    int firstGlobalRank = mpiRank * numGpusPerMpiRank;
    int lastGlobalRank = firstGlobalRank + numGpusPerMpiRank - 1;

    // Figure out the host and get the localGpuOffset
    int nameLen;
    char name[MPI_MAX_PROCESSOR_NAME];
    std::vector<char> allnames(numMpiRanks * MPI_MAX_PROCESSOR_NAME, 0);

    MPI_Get_processor_name(name, &nameLen);
    MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                    allnames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);

    for (int rank = 0; rank < mpiRank; rank++)
    {
        if (!strcmp(name, allnames.data() + (rank * MPI_MAX_PROCESSOR_NAME)))
            localGpuOffset += numGpusPerMpiRank;
    }

    printf("Rank %d [%s] LocalGpuOffset: %d GlobalRankFirst %d GlobalRankLast %d\n",
            mpiRank, name, localGpuOffset, firstGlobalRank, lastGlobalRank);
    
    // Create a unique ID and broadcast it to all ranks
    ncclUniqueId uniqueId;
    if (mpiRank == 0) ncclGetUniqueId(&uniqueId);
    MPI_Bcast(&uniqueId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Each rank has it's own comm and stream
    std::vector<ncclComm_t> comms(numGpusPerMpiRank);
    std::vector<hipStream_t> streams(numGpusPerMpiRank);
    
    // Initialize comms and strams
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < numGpusPerMpiRank; i++) {
        HIPCALL(hipSetDevice(localGpuOffset + i));
        NCCLCHECK(ncclCommInitRank(&(comms[i]), numGlobalRanks, uniqueId, firstGlobalRank + i));
        HIPCALL(hipStreamCreate(&(streams[i])));
    }
    NCCLCHECK(ncclGroupEnd());
    
    int numSkippedCalls = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (auto groupCall : groupCalls)
        if (groupCall.isValid)
            ReplayRccl(groupCall, comms, streams, localGpuOffset, numGpusPerMpiRank, firstGlobalRank, numGlobalRanks);
        else {
            if (mpiRank == 0) printf("[ERROR] in group call: (skipping...)\n");
            for (auto rd : groupCall.rankData) {
                if (mpiRank == 0) printf("  - Rank %02d: comm %s in line %d\n", rd.first, rd.second.comm.c_str(), rd.second.lineNum);
                for (int task = 0; task < rd.second.tasks.size(); task++) {
                    TaskInfo ti = rd.second.tasks[task];
                    if (mpiRank == 0)
                        printf("  - Task %02d: %32s inPlace=%d count=%lu datatype=%d op=%d root=%d\n",
                                task, ncclFuncNames[ti.funcType], ti.inPlace, ti.count, ti.datatype, ti.op, ti.root);
                }
            }
            numSkippedCalls++;
        }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Need to destroy comms and streams after collective execution is done
    for (int i = 0; i < numGpusPerMpiRank; ++i) {
        ncclCommDestroy(comms[i]);
        HIPCALL(hipStreamDestroy(streams[i]));
    }

    MPI_Finalize();

    if (mpiRank == 0) printf("Executed group calls: %zu\n", groupCalls.size() - numSkippedCalls);
    if (mpiRank == 0) printf("Skipped group calls: %d\n", numSkippedCalls);

    // Time it takes to execute all the group calls
    if (mpiRank == 0) printf("Execution Time: %f seconds\n", duration.count());

    // Means no hang
    printf("MPI Rank %d Success\n", mpiRank);
    
    return 0;
}
