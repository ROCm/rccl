#pragma once
#include <map>
#include <cstring>

#include <rccl/rccl.h>

// NOTE: Parsing is based on this line logging collective information in enqueue.cc
// INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d \
                       root %d comm %p [nranks=%d] stream %p task %d globalrank %d",
//                info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
//                info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream,
//                info->comm->tasks.nTasksP2p + info->comm->tasks.nTasksColl,
//                info->comm->localRankToRank[info->comm->localRank]);

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define HIPCALL(cmd)                                                                          \
    do {                                                                                      \
        hipError_t error = (cmd);                                                             \
        if (error != hipSuccess)                                                              \
        {                                                                                     \
            printf("Encountered HIP error (%s) at line %d in file %s\n",                      \
                                  hipGetErrorString(error), __LINE__, __FILE__);              \
            exit(-1);                                                                         \
        }                                                                                     \
    } while (0)

#define NCCLCHECK(cmd) do {                                     \
    ncclResult_t res = cmd;                                     \
    if (res != ncclSuccess) {                                   \
         printf("NCCL failure %s:%d '%s'\n",                    \
            __FILE__,__LINE__,ncclGetErrorString(res));         \
    }                                                           \
} while(0)

struct LineItem
{
  char   hostname[MPI_MAX_PROCESSOR_NAME];
  int    pid;
  int    tid;
  int    cudaDev;
  char   opName[32];
  int    opCount;
  char   sendbuff[32];
  char   recvbuff[32];
  size_t count;
  int    datatype;
  int    op;
  int    root;
  char   comm[32];
  int    nRanks;
  void*  stream;
  int    task;
  int    globalRank;
};

// Enumeration of all collective functions currently supported
typedef enum
{
  ncclCollBroadcast = 0,
  ncclCollReduce,
  ncclCollAllGather,
  ncclCollReduceScatter,
  ncclCollAllReduce,
  ncclCollGather,
  ncclCollScatter,
  ncclCollAllToAll,
  ncclCollAllToAllv,
  ncclCollSend,
  ncclCollRecv,
  ncclNumFuncs
} ncclFunc_t;

char const ncclFuncNames[ncclNumFuncs][32] =
{
  "Broadcast",
  "Reduce",
  "AllGather",
  "ReduceScatter",
  "AllReduce",
  "Gather",
  "Scatter",
  "AllToAll",
  "AllToAllv",
  "Send",
  "Recv"
};

struct TaskInfo
{
  ncclFunc_t     funcType;
  bool           inPlace;
  size_t         count;
  ncclDataType_t datatype;
  ncclRedOp_t    op;
  int            root;
};

struct RankData
{
  int                   lineNum;
  std::string           comm;
  std::vector<TaskInfo> tasks;
};

struct GroupCall
{
  bool isValid;
  int opCount;
  std::map<int, RankData> rankData;  // Indexed by globalRank
};

size_t DataTypeToBytes(ncclDataType_t const dataType)
{
  switch (dataType) {
  case ncclInt8:   return 1;
  case ncclUint8:  return 1;
  case ncclInt32:  return 4;
  case ncclUint32: return 4;
  case ncclInt64:  return 8;
  case ncclUint64: return 8;
  case ncclFloat16: return 2;
  case ncclFloat32: return 4;
  case ncclFloat64: return 8;
  case ncclBfloat16: return 2;
  default:
    printf("Unsupported datatype (%d)\n", dataType);
    exit(0);
  }
}

ncclFunc_t GetFuncType(char* func)
{
  for (int i = 0; i < ncclNumFuncs; i++)
    if (!strcmp(func, ncclFuncNames[i])) return (ncclFunc_t)i;
  printf("[ERROR] Unrecognzied func %s\n", func);
  exit(1);
}

// parse the logs and assign them into lineItem
bool ParseLineItem(char const* line, LineItem& li);

// this covers grouping the logs based on opCount and task number, 
// validatation of the groupCalls for both non-send/recv collectives and send/recv
void ParseCollectives(char const* logFilename,
                      int  const  numGlobalRanks,
                      std::vector<GroupCall>& groupCalls);

// size differ for each collective call and getSize gives a specific size in bytes depending on type of task, 
// global rank, element count and data type
std::pair<size_t, size_t> GetSize(TaskInfo taskInfo, 
                                  int numGlobalRanks);

// executes the collective call (task) 
void ExecuteCollective(TaskInfo task, ncclComm_t comm, hipStream_t stream, const void *sendbuff, void *recvbuff);

// allocates send/recv buff, sets the device based on which rank the task belongs to, 
// syncronize devices after executing all the tasks and free device memory. 
void ReplayRccl(GroupCall& groupCall, std::vector<ncclComm_t> comms, std::vector<hipStream_t> streams,
                                                                            int const localGpuOffset,
                                                                            int const numGpusPerMpiRank,
                                                                            int const firstGlobalRank,
                                                                            int const numGlobalRanks);