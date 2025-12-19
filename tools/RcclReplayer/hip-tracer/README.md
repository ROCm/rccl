# RCCL HIP Tracer Plugin

Captures HIP synchronization calls and records them into RCCL's trace files alongside NCCL/RCCL collective operations.

## HIP Calls Captured

- `hipStreamSynchronize`
- `hipDeviceSynchronize`
- `hipEventSynchronize`
- `hipEventRecord`
- `hipStreamWaitEvent`
- `hipEventCreate`
- `hipEventCreateWithFlags`
- `hipEventDestroy`

## Building

```bash
cd tools/RcclReplayer/hip-tracer/
make
```

This generates `librccl-hip-tracer.so`.

## Usage

Set environment variables and run your application:

```bash
export RCCL_REPLAY_FILE=/path/to/trace.json
export RCCL_HIP_TRACER_PLUGIN=/path/to/librccl-hip-tracer.so

python ml_app.py
```

RCCL automatically loads the plugin when `Recorder` initializes.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RCCL_REPLAY_FILE` | Path to output trace file (required) |
| `RCCL_HIP_TRACER_PLUGIN` | Path to `librccl-hip-tracer.so` |

## Output

HIP calls appear in the same trace file as RCCL operations:

```json
{
  CommInitRank : [...],
  HipStreamSynchronize : [stream : 0x7f123456, event : (nil), context : [...]],
  HipStreamWaitEvent : [stream : 0x7f123456, event : 0x7f789abc, context : [...]],
  HipEventRecord : [stream : 0x7f123456, event : 0x7f789abc, context : [...]],
  HipEventCreate : [stream : (nil), event : 0x7f789abc, context : [...]],
  HipEventDestroy : [stream : (nil), event : 0x7f789abc, context : [...]],
  AllReduce : [...],
}
```
