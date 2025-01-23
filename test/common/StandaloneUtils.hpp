#ifndef STANDALONE_UTILS_H
#define STANDALONE_UTILS_H

#include <iostream>
#include <cstdio>
#include <regex>
#include <vector>
#include <gtest/gtest.h>

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

// should be 112, temp fix to make CI pass
#define MAX_STACK_SIZE 448

#ifdef ENABLE_LL128
#define MAX_STACK_SIZE_gfx90a 296
#else
#define MAX_STACK_SIZE_gfx90a MAX_STACK_SIZE
#endif

struct KernelInfo {
    std::string name;
    int privateSegmentFixedSize = 0;
};

struct ArchInfo {
    std::string archName;
    std::vector<KernelInfo> kernels;
};

std::string executeCommand(const char* cmd);

std::vector<std::string> splitString(const std::string& str, char delimiter);


ArchInfo parseMetadata(const std::vector<std::string>& list);

namespace RcclUnitTesting
{
    void fork_and_launch_rccl(int nranks, int collID, const std::vector<int>& sendBuff, std::vector<int>& recvBuff, const std::vector<int>& expected, bool use_managed_mem = false);
    void call_rccl(ncclUniqueId id, int collID, int rank, int nranks, const std::vector<int>& send, std::vector<int>& recv, bool use_managed_mem = false);

}
#endif