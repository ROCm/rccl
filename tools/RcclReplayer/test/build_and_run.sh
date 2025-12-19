#!/bin/bash
# Build and run multi-thread single-GPU test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use same defaults as Makefile
ROCM_DIR=${ROCM_DIR:-/opt/rocm}
RCCL_DIR=${RCCL_DIR:-../../../build/release}
MPI_DIR=${MPI_DIR:-/opt/ompi}

# Build
echo "Building test..."
make clean
make

# Setup environment
export PATH=$MPI_DIR/bin:$PATH
export LD_LIBRARY_PATH=$RCCL_DIR:$MPI_DIR/lib:$ROCM_DIR/lib:$LD_LIBRARY_PATH
export RCCL_REPLAY_FILE=$SCRIPT_DIR/test_log.json
export RCCL_HIP_TRACER_PLUGIN=$SCRIPT_DIR/../hip-tracer/librccl-hip-tracer.so

# Configuration
MPI_RANKS=${1:-2}
NUM_THREADS=${2:-3}

echo ""
echo "=== Multi-Thread Single-GPU Test ==="
echo "  MPI ranks (GPUs): $MPI_RANKS"
echo "  Threads per GPU: $NUM_THREADS"
echo "  Total communicators: $NUM_THREADS"
echo "  Log prefix: $RCCL_REPLAY_FILE"
echo "====================================="
echo ""

# Run test
$MPI_DIR/bin/mpirun -np $MPI_RANKS ./test_multi_thread_single_gpu $NUM_THREADS

echo ""
echo "Test complete!"
echo "Logs:"
ls -la "$SCRIPT_DIR"/test_log.* 2>/dev/null || echo "No logs generated"
