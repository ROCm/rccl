/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "EnvVars.hpp"
#include "CollectiveArgs.hpp"
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>

namespace RcclUnitTesting
{
  int const UT_SINGLE_PROCESS = (1<<0);
  int const UT_MULTI_PROCESS  = (1<<1);

  int getDeviceCount(int *devices)
  {
    // Prepare parent->child pipe
    int pipefd[2];
    if (pipe(pipefd) == -1)
    {
      ERROR("Unable to create parent->child pipe for getting number of devices\n");
      return TEST_FAIL;
    }
    pid_t pid = fork();
    if (0 == pid)
    {
      int dev;
      hipGetDeviceCount(&dev);
      if (write(pipefd[1], &dev, sizeof(dev)) != sizeof(dev)) return TEST_FAIL;
      close(pipefd[0]);
      close(pipefd[1]);
      exit(EXIT_SUCCESS);
    }
    else
    {
      int status;
      if (read(pipefd[0], devices, sizeof(*devices)) != sizeof(*devices)) return TEST_FAIL;
      waitpid(pid, &status, 0);
      assert(!status);
      close(pipefd[0]);
      close(pipefd[1]);
    }
    return TEST_SUCCESS;
  }

  EnvVars::EnvVars()
  {
    // Collect number of GPUs available
    // NOTE: Cannot use HIP call prior to launching unless it is inside another child process
    numDetectedGpus = 0;
    getDeviceCount(&numDetectedGpus);

    showNames      = GetEnvVar("UT_SHOW_NAMES"  , 1);
    minGpus        = GetEnvVar("UT_MIN_GPUS"    , 2);
    maxGpus        = GetEnvVar("UT_MAX_GPUS"    , numDetectedGpus);
    onlyPow2Gpus   = GetEnvVar("UT_POW2_GPUS"   , false);
    processMask    = GetEnvVar("UT_PROCESS_MASK", UT_SINGLE_PROCESS | UT_MULTI_PROCESS);
    verbose        = GetEnvVar("UT_VERBOSE"     , 0);
    printValues    = GetEnvVar("UT_PRINT_VALUES", 0);
    maxRanksPerGpu = GetEnvVar("UT_MAX_RANKS_PER_GPU", 1);
    showTiming     = GetEnvVar("UT_SHOW_TIMING",  1);
    useInteractive = GetEnvVar("UT_INTERACTIVE",  0);

    // Limit number of supported reduction operators to just ncclSum if only allReduce is built
#ifdef BUILD_ALLREDUCE_ONLY
    int numOps = 1;
#else
    int numOps = ncclNumOps;
#endif
    std::vector<std::string> redOpStrings = GetEnvVarsList("UT_REDOPS");
    for (auto s : redOpStrings)
    {
      for (int i = 0; i < numOps; ++i)
      {
        if (!strcmp(s.c_str(), ncclRedOpNames[i]))
        {
          redOps.push_back((ncclRedOp_t)i);
          break;
        }
      }
    }
    // Default back to all ops if no strings are found
    if (redOps.empty())
    {
      for (int i = 0; i < numOps; i++)
        redOps.push_back((ncclRedOp_t)i);
    }

    // Limit number of supported datatypes if only allReduce is built
    std::vector<std::string> dtStrings = GetEnvVarsList("UT_DATATYPES");
    for (auto s : dtStrings)
    {
      for (int i = 0; i < ncclNumTypes; ++i)
      {
        if (!strcmp(s.c_str(), ncclDataTypeNames[i]))
        {
#ifdef BUILD_ALLREDUCE_ONLY
          if (i == ncclFloat32)
#endif
          {
            dataTypes.push_back((ncclDataType_t)i);
          }
        }
      }
    }

    // Default option if no valid datatypes are found in env var
    if (dataTypes.empty())
    {
      dataTypes.push_back(ncclFloat32);
      // Skip all but 32-bit floats if only AllReduce is being built
#ifndef BUILD_ALLREDUCE_ONLY
      dataTypes.push_back(ncclInt8);
      dataTypes.push_back(ncclUint8);
      dataTypes.push_back(ncclInt32);
      dataTypes.push_back(ncclUint32);
      dataTypes.push_back(ncclInt64);
      dataTypes.push_back(ncclUint64);
      dataTypes.push_back(ncclFloat16);
      dataTypes.push_back(ncclFloat32);
      dataTypes.push_back(ncclFloat64);
      dataTypes.push_back(ncclBfloat16);
#endif
    }

    // Build list of possible # GPU ranks based on env vars
    numGpusList.clear();
    for (int i = minGpus; i <= maxGpus; i++)
      if (!onlyPow2Gpus || ((i & (i-1)) == 0))
        numGpusList.push_back(i);

    // Build isMultiProcessList
    isMultiProcessList.clear();
    if (this->processMask & UT_SINGLE_PROCESS) isMultiProcessList.push_back(0);
    if (this->processMask & UT_MULTI_PROCESS)  isMultiProcessList.push_back(1);
  }

  std::vector<ncclRedOp_t> const& EnvVars::GetAllSupportedRedOps()
  {
    return redOps;
  }

  std::vector<ncclDataType_t> const& EnvVars::GetAllSupportedDataTypes()
  {
    return dataTypes;
  }

  std::vector<int> const& EnvVars::GetNumGpusList()
  {
    return numGpusList;
  }

  std::vector<int> const& EnvVars::GetIsMultiProcessList()
  {
    return isMultiProcessList;
  }

  int EnvVars::GetEnvVar(std::string const varname, int defaultValue)
  {
    if (getenv(varname.c_str()))
      return atoi(getenv(varname.c_str()));
    return defaultValue;
  };

  std::vector<std::string> EnvVars::GetEnvVarsList(std::string const varname)
  {
    std::vector<std::string> result;
    if (getenv(varname.c_str()))
    {
      char* token = strtok(getenv(varname.c_str()), ",;");
      while (token != NULL)
      {
        result.push_back(token);
        token = strtok(NULL, ",;");
      }
    }
    return result;
  }

  void EnvVars::ShowConfig()
  {
    std::vector<std::tuple<std::string, int, std::string>> supported =
      {
        std::make_tuple("UT_SHOW_NAMES"       , showNames     , "Show test case names"),
        std::make_tuple("UT_MIN_GPUS"         , minGpus       , "Minimum number of GPUs to use"),
        std::make_tuple("UT_MAX_GPUS"         , maxGpus       , "Maximum number of GPUs to use"),
        std::make_tuple("UT_POW2_GPUS"        , onlyPow2Gpus  , "Only allow power-of-2 # of GPUs"),
        std::make_tuple("UT_PROCESS_MASK"     , processMask   , "Whether to run single/multi process"),
        std::make_tuple("UT_VERBOSE"          , verbose       , "Show verbose unit test output"),
        std::make_tuple("UT_REDOPS"           , -1            , "List of reduction ops to test"),
        std::make_tuple("UT_DATATYPES"        , -1            , "List of datatypes to test"),
        std::make_tuple("UT_MAX_RANKS_PER_GPU", maxRanksPerGpu, "Maximum number of ranks using the same GPU"),
        std::make_tuple("UT_PRINT_VALUES"     , printValues   , "Print array values (-1 for all)"),
        std::make_tuple("UT_SHOW_TIMING"      , showTiming    , "Show timing table"),
        std::make_tuple("UT_INTERACTIVE"      , useInteractive, "Run in interactive mode")
      };

    printf("================================================================================\n");
    printf(" Environment variables:\n");
    for (auto p : supported)
    {
      printf(" - %-20s %-42s (%3d) %s\n", std::get<0>(p).c_str(), std::get<2>(p).c_str(), std::get<1>(p),
             getenv(std::get<0>(p).c_str()) ? getenv(std::get<0>(p).c_str()) : "<unset>");
    }
    printf("================================================================================\n");
  }
}
