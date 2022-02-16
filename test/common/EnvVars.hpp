/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once
#include <hsa/hsa.h>
#include <vector>
#include "rccl.h"

namespace RcclUnitTesting
{
  // Helper function to count the number of GPUs on system
  static hsa_status_t CountGpus(hsa_agent_t agent, void* data);

  // Helper class to track environment variables that affect the unit tests
  class EnvVars
  {
  public:
    bool showNames;   // List test case names during run        [UT_SHOW_NAMES]
    int  minGpus;     // Set the minimum number of GPUs to use  [UT_MIN_GPUS]
    int  maxGpus;     // Set the maximum number of GPUs to use  [UT_MAX_GPUS]
    int  processMask; // Filter single/multi process            [UT_PROCESS_MASK]
    bool verbose;     // Show verbose TestBed output for debug  [UT_VERBOSE]
    int  printValues; // Print out input/output/expected arrays [UT_PRINT_VALUES]

    // Constructor that parses and collects environment variables
    EnvVars();

    std::vector<ncclRedOp_t>    const& GetAllSupportedRedOps();
    std::vector<ncclDataType_t> const& GetAllSupportedDataTypes();

    static void ShowConfig();

  protected:
    std::vector<ncclRedOp_t>    redOps;    // Supported reduction ops [UT_REDOPS]
    std::vector<ncclDataType_t> dataTypes; // Support datatypes       [UT_DATATYPES]

    // Helper functions to parse environment variables
    int GetEnvVar(std::string const varname, int defaultValue);
    std::vector<std::string> GetEnvVarsList(std::string const varname);
  };
}
