#ifndef HELLORCCL_HPP
#define HELLORCCL_HPP
#include <iostream>

#define HIP_CALL(cmd)                                                 \
  do {                                                                \
    hipError_t error = (cmd);                                         \
    if (error != hipSuccess)                                          \
    {                                                                   \
      std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " \
                << __LINE__ << " in file " << __FILE__ << "\n";         \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#define NCCL_CALL(cmd) \
  do { \
    ncclResult_t error = (cmd);                 \
    if (error != ncclSuccess)                   \
    {                                           \
      std::cerr << "Encountered NCCL error (" << ncclGetErrorString(error) << ") at line " \
                << __LINE__ << " in file " << __FILE__ << "\n";         \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#endif
