#ifndef ENVVARS_HPP
#define ENVVARS_HPP

// This class manages environment variable that affect TransferBench
class EnvVars
{
public:
  // Default configuration values
  int const DEFAULT_NUM_WARMUPS     =  3;
  int const DEFAULT_NUM_ITERATIONS  = 10;
  int const DEFAULT_SAMPLING_FACTOR =  1;

  // Environment variables
  int useHipCall;      // Use hipMemcpy/hipMemset instead of custom shader kernels
  int useMemset;       // Perform a memset instead of a copy (ignores source memory)
  int useFineGrainMem; // Allocate fine-grained GPU memory instead of coarse-grained GPU memory
  int useSingleSync;   // Perform synchronization only once after all iterations instead of per iteration
  int useInteractive;  // Pause for user-input before starting transfer loop
  int useSleep;        // Adds a 100ms sleep after each synchronization
  int combineTiming;   // Combines the timing with kernel launch
  int showAddr;        // Print out memory addresses for each Link
  int outputToCsv;     // Output in CSV format
  int byteOffset;      // Byte-offset for memory allocations
  int numWarmups;      // Number of un-timed warmup iterations to perform
  int numIterations;   // Number of timed iterations to perform
  int samplingFactor;  // Affects how many different values of N are generated (when N set to 0)


  // Constructor that collects values
  EnvVars()
  {
    useHipCall      = GetEnvVar("USE_HIP_CALL"     , 0);
    useMemset       = GetEnvVar("USE_MEMSET"       , 0);
    useFineGrainMem = GetEnvVar("USE_FINEGRAIN_MEM", 0);
    useSingleSync   = GetEnvVar("USE_SINGLE_SYNC"  , 0);
    useInteractive  = GetEnvVar("USE_INTERACTIVE"  , 0);
    combineTiming   = GetEnvVar("COMBINE_TIMING"   , 0);
    showAddr        = GetEnvVar("SHOW_ADDR"        , 0);
    outputToCsv     = GetEnvVar("OUTPUT_TO_CSV"    , 0);
    byteOffset      = GetEnvVar("BYTE_OFFSET"      , 0);
    numWarmups      = GetEnvVar("NUM_WARMUPS"      , DEFAULT_NUM_WARMUPS);
    numIterations   = GetEnvVar("NUM_ITERATIONS"   , DEFAULT_NUM_ITERATIONS);
    samplingFactor  = GetEnvVar("SAMPLING_FACTOR"  , DEFAULT_SAMPLING_FACTOR);

    // Perform some basic validation
    if (byteOffset % sizeof(float))
    {
      printf("[ERROR] BYTE_OFFSET must be set to multiple of %lu\n", sizeof(float));
      exit(1);
    }
    if (numWarmups < 0)
    {
      printf("[ERROR] NUM_WARMUPS must be set to a non-negative number\n");
      exit(1);
    }
    if (numIterations <= 0)
    {
      printf("[ERROR] NUM_ITERATIONS must be set to a positive number\n");
      exit(1);
    }
    if (samplingFactor < 1)
    {
      printf("[ERROR] SAMPLING_FACTOR must be greater or equal to 1\n");
      exit(1);
    }
  }

  // Display info on the env vars that can be used
  static void DisplayUsage()
  {
    printf("Environment variables:\n");
    printf("======================\n");
    printf(" USE_HIP_CALL       - Use hipMemcpy/hipMemset instead of custom shader kernels\n");
    printf(" USE_MEMSET         - Perform a memset instead of a copy (ignores source memory)\n");
    printf(" USE_FINEGRAIN_MEM  - Allocate fine-grained GPU memory instead of coarse-grained GPU memory\n");
    printf(" USE_SINGLE_SYNC    - Perform synchronization only once after all iterations instead of per iteration\n");
    printf(" USE_INTERACTIVE    - Pause for user-input before starting transfer loop\n");
    printf(" COMBINE_TIMING     - Combines timing with launch (potentially lower timing overhead)\n");
    printf(" SHOW_ADDR          - Print out memory addresses for each Link\n");
    printf(" OUTPUT_TO_CSV      - Outputs to CSV format if set\n");
    printf(" BYTE_OFFSET        - Initial byte-offset for memory allocations.  Must be multiple of 4. Defaults to 0\n");
    printf(" NUM_WARMUPS=W      - Perform W untimed warmup iteration(s) per test\n");
    printf(" NUM_ITERATIONS=I   - Perform I timed iteration(s) per test\n");
    printf(" SAMPLING_FACTOR=F  - Add F samples (when possible) between powers of 2 when auto-generating data sizes\n");
  }

  // Display env var settings
  void DisplayEnvVars() const
  {
    if (!outputToCsv)
    {
      printf("Run configuration\n");
      printf("=====================================================\n");
      printf("%-20s: Using %s\n", "USE_HIP_CALL",
             useHipCall ? "HIP functions" : "custom kernels");
      printf("%-20s: Performing %s\n", "USE_MEMSET",
             useMemset ? "memset" : "memcopy");
      if (useHipCall && !useMemset)
      {
        char* env = getenv("HSA_ENABLE_SDMA");
        printf("%-20s: %s\n", "HSA_ENABLE_SDMA",
               (env && !strcmp(env, "0")) ? "Using blit kernels for hipMemcpy" : "Using DMA copy engines");
      }
      printf("%-20s: GPU destination memory type: %s-grained\n", "USE_FINEGRAIN_MEM",
             useFineGrainMem ? "fine" : "coarse");
      printf("%-20s: %s\n", "USE_SINGLE_SYNC",
             useSingleSync ? "Synchronizing only once, after all iterations" : "Synchronizing per iteration");
      printf("%-20s: Running in %s mode\n", "USE_INTERACTIVE",
             useInteractive ? "interactive" : "non-interactive");
      printf("%-20s: %s\n", "COMBINE_TIMING",
             combineTiming ? "Using combined timing+launch" : "Using separate timing / launch");
      printf("%-20s: %s\n", "SHOW_ADDR",
             showAddr ? "Displaying src/dst mem addresses" : "Not displaying src/dst mem addresses");
      printf("%-20s: Output to %s\n", "OUTPUT_TO_CSV",
             outputToCsv ? "CSV" : "console");
      printf("%-20s: Using byte offset of %d\n", "BYTE_OFFSET", byteOffset);
      printf("%-20s: Running %d warmup iteration(s) per topology\n", "NUM_WARMUPS", numWarmups);
      printf("%-20s: Running %d timed iteration(s) per topology\n", "NUM_ITERATIONS", numIterations);
      printf("\n");
    }
  };

private:
  // Helper function that gets parses environment variable or sets to default value
  int GetEnvVar(std::string const varname, int defaultValue)
  {
    if (getenv(varname.c_str()))
      return atoi(getenv(varname.c_str()));
    return defaultValue;
  }
};

#endif
