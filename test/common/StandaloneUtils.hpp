#ifndef STANDALONE_UTILS_H
#define STANDALONE_UTILS_H

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

#endif