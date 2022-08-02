# Changelog for TransferBench

## v1.02
### Added
- Setting NUM_ITERATIONS to negative number indicates to run for -NUM_ITERATIONS seconds per Test
### Changed
- Copies are now refered to as Transfers instead of Links
- Re-ordering how env vars are displayed (alphabetically now)
### Removed
- Combined timing is now always on for kernel-based GPU copies. COMBINED_TIMING env var has been removed
- Use single sync is no longer supported to facility variable iterations. USE_SINGLE_SYNC env var has been removed

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
