# Change Log for RCCL

Full documentation for RCCL is available at [https://rccl.readthedocs.io](https://rccl.readthedocs.io)

## (Unreleased) RCCL-2.12.10
### Added
- Compatibility with NCCL 2.12.10
- Packages for test and benchmark executables on all supported OSes using CPack.
- Adding custom signal handler - opt-in with RCCL_ENABLE_SIGNALHANDLER=1
  - Additional details provided if Binary File Descriptor library (BFD) is pre-installed
- Adding experimental support for using multiple ranks per device
  - Requires using a new interface to create communicator (ncclCommInitRankMulti), please
    refer to the interface documentation for details.
  - To avoid potential deadlocks, user might have to set an environment variables increasing
    the number of hardware queues (e.g. export GPU_MAX_HW_QUEUES=16)
- Adding support for reusing ports in NET/IB channels
  - Opt-in with NCCL_IB_SOCK_CLIENT_PORT_REUSE=1 and NCCL_IB_SOCK_SERVER_PORT_REUSE=1
  - When "Call to bind failed : Address already in use" error happens in large-scale AlltoAll
    (e.g., >=64 MI200 nodes), users are suggested to opt-in either one or both of the options
    to resolve the massive port usage issue
  - Avoid using NCCL_IB_SOCK_SERVER_PORT_REUSE when NCCL_NCHANNELS_PER_NET_PEER is tuned >1
### Removed
- Removed experimental clique-based kernels

## RCCL-2.11.4 for ROCm 5.1.0
### Added
- Compatibility with NCCL 2.11.4
### Known issues
- Managed memory is not currently supported for clique-based kernels

## RCCL-2.10.3 for ROCm 5.0.0
### Added
- Compatibility with NCCL 2.10.3
### Known issues
- Managed memory is not currently supported for clique-based kernels

## RCCL-2.9.9 for ROCm 4.5.0
### Changed
- Packaging split into a runtime package called rccl and a development package called rccl-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.
### Added
- Compatibility with NCCL 2.9.9
### Known issues
- Managed memory is not currently supported for clique-based kernels

## [RCCL-2.8.4 for ROCm 4.3.0]
### Added
- Ability to select the number of channels to use for clique-based all reduce (RCCL_CLIQUE_ALLREDUCE_NCHANNELS).  This can be adjusted to tune for performance when computation kernels are being executed in parallel.
### Optimizations
- Additional tuning for clique-based kernel AllReduce performance (still requires opt in with RCCL_ENABLE_CLIQUE=1)
- Modification of default values for number of channels / byte limits for clique-based all reduce based on device architecture
### Changed
- Replaced RCCL_FORCE_ENABLE_CLIQUE to RCCL_CLIQUE_IGNORE_TOPO
- Clique-based kernels can now be enabled on topologies where all active GPUs are XGMI-connected
- Topologies not normally supported by clique-based kernels require RCCL_CLIQUE_IGNORE_TOPO=1
### Fixed
- Install script '-r' flag invoked alone no longer incorrectly deletes any existing builds.
### Known issues
- Managed memory is not currently supported for clique-based kernels

## [RCCL-2.8.4 for ROCm 4.2.0]
### Added
- Compatibility with NCCL 2.8.4

### Optimizations
- Additional tuning for clique-based kernels
- Enabling GPU direct RDMA read from GPU
- Fixing potential memory leak issue when re-creating multiple communicators within same process
- Improved topology detection
### Known issues
- None

## [RCCL-2.7.8 for ROCm 4.1.0]
### Added
- Experimental support for clique-based kernels (opt in with RCCL_ENABLE_CLIQUE=1)
- Clique-based kernels may offer better performance for smaller input sizes
- Clique-based kernels are currently only enabled for AllReduce under a certain byte limit (controlled via RCCL_CLIQUE_ALLREDUCE_BYTE_LIMIT)
### Optimizations
- Performance improvements for Rome-based systems
### Known issues
- Clique-based kernels are currently experimental and have not been fully tested on all topologies.  By default, clique-based kernels are disabled if the detected topology is not supported (override with RCCL_FORCE_ENABLE_CLIQUE)
- Clique-based kernels may hang if there are differences between environment variables set across ranks.
- Clique-based kernels may fail if the input / output device pointers are not the base device pointers returned by hipMalloc.


## [RCCL-2.7.8 for ROCm 3.9.0]
### Added
- Adding support for alltoallv RCCL kernel
### Optimizations
- Modifications to topology based on XGMI links
### Known issues
- None

## [RCCL-2.7.6 for ROCm 3.8.0]
### Added
- Support for static library builds
### Known issues
- None

## [RCCL-2.7.6 for ROCm 3.7.0]
### Added
- Updated to RCCL API version of 2.7.6
- Added gather, scatter and all-to-all collectives

## [RCCL-2.7.0 for ROCm 3.6.0]
### Added
- Updated to RCCL API version of 2.6.4

## [RCCL-2.7.0 for ROCm 3.5.0]
### Added
- Compatibility with NCCL 2.6
- Network interface improvements with API v3
### Optimizations
- Fixing issues and built time improvements for hip-clang
- Network topology detection
- Improved CPU type detection
- Infiniband adaptive routing support
### Changed
- Switched to hip-clang as default compiler
### Deprecated
- Deprecated hcc build
