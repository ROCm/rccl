# Changelog for TransferBench

## v1.01
### Added
- Adding USE_SINGLE_STREAM feature
  - All Links that execute on the same GPU device are executed with a single kernel launch on a single stream
  - Does not work with USE_HIP_CALL and forces USE_SINGLE_SYNC to collect timings
  - Adding ability to request coherent / fine-grained host memory ('B')
### Changed
- Separating TransferBench from RCCL repo
- Peer-to-peer benchmark mode now works OUTPUT_TO_CSV
- Toplogy display now works with OUTPUT_TO_CSV
- Moving documentation about config file into example.cfg
### Removed
- Removed config file generation
- Removed show pointer address environment variable (SHOW_ADDR)
