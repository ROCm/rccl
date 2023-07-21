# RCCL REPLAYER
Collective log replayer and fault detection tool for RCCL.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Installation](#installation)
5. [Usage](#usage)

## Introduction

Replayer is a tool designed to analyze and replay collective logs obtained from RCCL (Rocm Collective Communications Library) runs. It helps detecting faulty group calls and skips them during the replay process. By analyzing the collected logs, Replayer assists in identifying potential issues in collective communication patterns, enabling you to optimize and improve the performance of your MPI-based applications.

## Features

- Collects and aggregates collective logs from RCCL runs.
- Detects missing/faulty group calls and provides report.
- Replays collective calls based on the recorded data.
- Allows skipping of faulty group calls during replay.
- Supports various MPI ranks and GPU configurations.
- Supports multi-node environment. 

## How It Works

Replayer operates in the following steps:

1. **Collective Log Collection:** During your RCCL runs, the collective logs are generated, capturing important information like hostname, deviceIdx, collective call type, number of elements used, data type, operation type, task number, and global rank number about collective communication patterns.

2. **Data Aggregation:** Replayer collects and pareses the collective logs. organizing them based on opCount (collective count in the group call), and global rank information.

3. **Group Call Validation:** After acquiring data from the collective logs and generating group calls, the replayer validates the results using two different methods. For Non-Send/Recv collectives, it checks if each MPI rank has the required number of collective tasks. For Send/Recv collectives, it verifies if they all have a matching pair.

4. **Replaying RCCL:** Based on the aggregated and validated data, Replayer can replay the collective logs to reproduce the RCCL runs from your application.

5. **Reporting and Skipping.** Replayer outputs the detected faulty group calls and skips them during replay. It provides a report showing which group calls were skipped and why and, at the end, summarizes how many group calls were replayed and how many were skipped.

*Note:*  Before using the replayer, it is essential to collect your application's logs with 'NCCL_DEBUG=INFO' and 'NCCL_DEBUG_SUBSYS=COLL' enabled. These environment variables will ensure that the necessary debugging information related to collective communication is captured and recorded in the logs.

## Installation

To build the replayer, go to the rccl_replayer directory and simply run 'make'.

```bash
    cd rccl/tools/rccl_replayer
    make MPI_DIR=/path/to/mpi
```

Depending on the MPI library used and your installation path, you may need to edit the Makefile and set the MPI_DIR path accordingly.


## Usage

After successfully building the replayer, you can run it using the following command:

```bash
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL mpirun -np <numProcesses> ./rcclReplayer </path/to/logfile> <numGpusPerMpiRank>
```

Replace <numProcesses> with the number of MPI processes you want to run during the replay, </path/to/logfile> with the path to the collective log file generated during your RCCL runs, and <numGpusPerMpiRank> with the number of GPUs per MPI rank used in your application.

Depending on the MPI library you use, you may need to modify the mpirun command accordingly. The flag NCCL_DEBUG_SUBSYS=COLL ensures that only collective log information is printed to the terminal.

### Multi-Node Environment:

If multiple nodes were used for your application, you can also replay the collective logs with using multiple nodes. See the following command:

```bash
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL mpirun --hostfile <path/to/hostfile.txt> -np <numProcesses> ./rcclReplayer </path/to/logfile> <numGpusPerMpiRank>
```

### SLURM:

For systems using SLURM, you can use the following command to replay the collective logs:

```bash
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL srun -N <numNodes> -n <numProcesses> ./rcclReplayer </path/to/logfile> <numGpusPerMpiRank>
```

Replace 'numNodes' with the number of nodes used in your application.
