## Testing RCCL

### Directory structure

- `conformance:` Tests check if different error status codes emitted are valid or not
- `performance:` Evaluates performance of rccl ops, command line args should be provided to test a configuration
- `stress:` Stress tests with data from 1 element count to specified by user as command line arg
- `validation:` Validates output from running rccl op. If the data gets validated, the test exits with code `0`

### Debugging

Use different environment variables to emit information about different function calls happening in the library.
```RCCL_TRACE_RT=2 # prints out what rccl apis gets called and their arguments```

```RCCL_TRACE_RT=4 # prints out different arguments passed to kernels in rccl```
