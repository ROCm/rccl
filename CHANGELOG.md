# Change Log for RCCL

Full documentation for RCCL is available at [https://rccl.readthedocs.io](https://rccl.readthedocs.io)

## [Unreleased]
### Optimizations
- Performance improvements for Rome-based systems

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
