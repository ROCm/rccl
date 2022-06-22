/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "EnvVars.hpp"
#include "CollectiveArgs.hpp"
#include <cstdlib>

namespace RcclUnitTesting
{
  int const UT_SINGLE_PROCESS = (1<<0);
  int const UT_MULTI_PROCESS  = (1<<1);

  hsa_status_t CountGpus(hsa_agent_t agent, void* data)
  {
    int* currCount = (int*)data;
    hsa_device_type_t device;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device);
    if (device == HSA_DEVICE_TYPE_GPU)
      *currCount = *currCount + 1;
    return HSA_STATUS_SUCCESS;
  }

  EnvVars::EnvVars()
  {
    // Collect number of GPUs available
    // NOTE: Cannot use HIP call prior to launching child processes via fork so use HSA
    int numDevicesAvailable = 0;
    hsa_init();
    hsa_iterate_agents(CountGpus, &numDevicesAvailable);
    hsa_shut_down();

    showNames      = GetEnvVar("UT_SHOW_NAMES"  , 1);
    minGpus        = GetEnvVar("UT_MIN_GPUS"    , 2);
    maxGpus        = GetEnvVar("UT_MAX_GPUS"    , numDevicesAvailable);
    processMask    = GetEnvVar("UT_PROCESS_MASK", UT_SINGLE_PROCESS | UT_MULTI_PROCESS);
    verbose        = GetEnvVar("UT_VERBOSE"     , 0);
    printValues    = GetEnvVar("UT_PRINT_VALUES", 0);
    maxRanksPerGpu = GetEnvVar("UT_MAX_RANKS_PER_GPU", 1);

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
      // Half-precision floats disabled due to lack of host-side support
      // dataTypes.push_back(ncclFloat16);
      dataTypes.push_back(ncclFloat32);
      dataTypes.push_back(ncclFloat64);
      dataTypes.push_back(ncclBfloat16);
#endif
    }
  }

  std::vector<ncclRedOp_t> const& EnvVars::GetAllSupportedRedOps()
  {
    return redOps;
  }

  std::vector<ncclDataType_t> const& EnvVars::GetAllSupportedDataTypes()
  {
    return dataTypes;
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
    std::vector<std::pair<std::string, std::string>> supported =
      {
        std::make_pair("UT_SHOW_NAMES"       , "Show test case names"),
        std::make_pair("UT_MIN_GPUS"         , "Minimum number of GPUs to use"),
        std::make_pair("UT_MAX_GPUS"         , "Maximum number of GPUs to use"),
        std::make_pair("UT_PROCESS_MASK"     , "Whether to run single/multi process"),
        std::make_pair("UT_VERBOSE"          , "Show verbose unit test output"),
        std::make_pair("UT_REDOPS"           , "List of reduction ops to test"),
        std::make_pair("UT_DATATYPES"        , "List of datatypes to test"),
        std::make_pair("UT_MAX_RANKS_PER_GPU", "Maximum number of ranks using the same GPU"),
        std::make_pair("UT_PRINT_VALUES"     , "Print array values (# of values to print, < 0 for all)")
      };

    printf("================================================================================\n");
    printf(" Environment variables:\n");
    for (auto p : supported)
    {
      printf(" - %-20s %-40s %s\n", p.first.c_str(), p.second.c_str(),
             getenv(p.first.c_str()) ? getenv(p.first.c_str()) : "<unset>");
    }
    printf("================================================================================\n");
  }
}
