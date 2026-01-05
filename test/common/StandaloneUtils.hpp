#ifndef STANDALONE_UTILS_H
#define STANDALONE_UTILS_H

#include <cstdio>
#include <vector>
#include <string>
#include <rccl/rccl.h>

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

// Stack size limits for generic kernels (BUILD_GENERIC_KERNELS=ON)
#define MAX_STACK_SIZE 640

// Self-contained kernels (BUILD_GENERIC_KERNELS=OFF) have higher stack usage
// because they inline all device functions. This is a trade-off for faster compilation.
#ifdef BUILD_GENERIC_KERNELS
#define MAX_STACK_SIZE_SELF_CONTAINED MAX_STACK_SIZE
#else
#define MAX_STACK_SIZE_SELF_CONTAINED 1100
#endif

#ifdef ENABLE_LL128
#define MAX_STACK_SIZE_gfx90a 360
#else
#define MAX_STACK_SIZE_gfx90a MAX_STACK_SIZE
#endif

namespace RcclUnitTesting
{
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
}
#endif
