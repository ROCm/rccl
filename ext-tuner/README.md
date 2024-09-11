# RCCL Tuner Plugin API 

This document describes the API structure to be implemented by an external tuner for RCCL. The purpose of this plugin is to enable stakeholders to select an algorithm, a protocol, number of channels (thread blocks) based on the input configuration: message size, number of nodes and GPUs, and link types (PCIe, XGMI, NET).

## Notes
- The [example plugin](example/plugin.c) is a demonstration that uses math models to approximate BW and latency of all available choices of algorithms and protocols and provide the one that scores the lowest latency.
- The API allows partial outputs: tuners can set only the algorithm and protocol, or let NCCL set the remaining fields (e.g., number of channels).
- If `getCollInfo()` fails, RCCL will use its default internal mechanisms to determine the best collective configuration.
- `getCollInfo()` is called for each collective call, so special care is to be taken not to cause excessive latency.
- The advantage of this plugin is that each customer can create and maintain their hand-tailored tuner without relying on RCCL to create and maintain it.
- Supported RCCL algorithms are `NCCL_ALGO_TREE` and `NCCL_ALGO_RING`.
- Supported RCCL protocols are `NCCL_PROTO_SIMPLE`, `NCCL_PROTO_LL` and `NCCL_PROTO_LL128`.
-- Until support is present, our example ignores other algorithms in `pluginGetCollInfo` API implementation
```
if ((a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) && collNetSupport != 1) continue;
if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && nvlsSupport != 1) continue;
if (a == NCCL_ALGO_NVLS && collNetSupport != 1) continue;
```
# API Description 
## Structure: `ncclTuner_v1_t`

### Fields

#### 1. `name`
  Type: `const char*`  
  Description: The name of the tuner. Can be used for logging purposes when (`NCCL_DEBUG=info NCCL_DEBUG_SUBSYS=tune`) are set.

### Functions

#### 1. `init`

Initializes the tuner states. Each communicator initializes its tuner. nNodes x nRanks = total number of GPUs participating in the collective communication

- **Parameters**:
  - `nRanks` (size_t): The number of ranks (GPUs) in the current communicator.
  - `nNodes` (size_t): The number of nodes (could be a mix of local and remote nodes).
  - `logFunction` (ncclDebugLogger_t): A log function that can be useful to turn on certain debugging info.

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


## Build instructions and usage

- The way to use the external plugin is to implement the desired algorithm/protocol selection technique using the API described above. `ext-tuner/example/plugin.c` is an example based on MI300 tuning table by default as a reference for customers in `plugin.c`.
- Then build the `libnccl-tuner.so` file with the Makefile provided in the same directory:

### Building and using example libnccl-tuner.so
```
cd $RCCL_HOME/ext-tuner/example/ 
make
```
Next is to let RCCL know that you want to use the custom-made libnccl-tuner.so by setting the following environment variable to the directory of the libnccl-tuner.so file:

```
export NCCL_TUNER_PLUGIN=$RCCL_HOME/ext-tuner/example/libnccl-tuner.so
```

