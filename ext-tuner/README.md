# RCCL Tuner Plugin API 

This document describes the API structure to be implemented by an external tuner for RCCL. The purpose of this plugin is to enable stakeholders to select an algorithm, a protocol, number of channels (thread blocks) based on the input configuration

## Notes
- The file plugin.c is an example that uses regression to approximate BW and latency of all choices and provide that one that scores lowest latency.
- The API allows partial outputs: tuners can set only the algorithm and protocol, or let NCCL set the remaining fields (e.g., number of channels).
- If `getCollInfo()` fails, NCCL will use its default internal mechanisms to determine the best collective configuration.
- COLLNET algorithms (`NCCL_ALGO_COLLNET_DIRECT` and `NCCL_ALGO_COLLNET_CHAIN`) are only supported when NVLink SHARP is present (support is provided as input from the RCCL library).
- Once the API is built, use 

# API Description 
## Structure: `ncclTuner_v1_t`

### Fields

#### 1. `name`
  Type: `const char*`  
  Description: The name of the tuner. Used for logging purposes when (`NCCL_DEBUG=info NCCL_DEBUG_SUBSYS=tune`) are set.

### Functions

#### 1. `init`

Initializes the tuner states.

- **Parameters**:
  - `nRanks` (size_t): The number of ranks in the current communicator. Each communicator initializes its own tuner.
  - `nNodes` (size_t): The number of nodes in the current communicator.
  - `logFunction` (ncclDebugLogger_t): A log function that can be useful to integrate logging together with the NCCL core.

- **Return**:  
  Type: `ncclResult_t`  
  The result of the initialization.

#### 2. `getCollInfo`

Retrieves information about the collective algorithm, protocol, and other details for a given operation.

- **Parameters**:
  - `collType` (ncclFunc_t): The collective type, e.g., `allreduce`, `allgather`, etc.
  - `nBytes` (size_t): The size of the collective in bytes.
  - `collNetSupport` (int): Whether `collNet` supports this type.
  - `nvlsSupport` (int): Whether NVLink SHARP supports this type.
  - `numPipeOps` (int): The number of operations in the group.
  
- **Outputs**:
  - `algorithm` (int*): The selected algorithm to be used for the given collective.
  - `protocol` (int*): The selected protocol to be used for the given collective.
  - `nChannels` (int*): The number of channels (and SMs) to be used.
  
- **Description**:
  If `getCollInfo()` does not return `ncclSuccess`, NCCL will fall back to its default tuning for the given collective. The tuner is allowed to leave fields unset, in which case NCCL will automatically set those fields.

- **Return**:  
  Type: `ncclResult_t`  
  The result of the operation.

#### 3. `destroy`

Terminates the plugin and cleans up any resources allocated by the tuner.

- **Return**:  
  Type: `ncclResult_t`  
  The result of the cleanup process.

---
