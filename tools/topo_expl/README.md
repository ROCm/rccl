# RCCL Topology Explorer (topo_expl)

The RCCL Topology Explorer is a tool for analyzing and exploring network topologies for RCCL (ROCm Communication Collectives Library) collective operations. It simulates various hardware configurations and displays the actual algo/proto combo selections that RCCL would make.

## Building

### Prerequisites
- ROCm/HIP development environment
- RCCL source code
- hipify-perl (for source transformation)

### Build Instructions

```bash
cd tools/topo_expl
make
```

## Usage

```bash
./topo_expl -m model_id [-n numNodes=1]
```

### Parameters

- `-m model_id`: Specifies the topology model to use (required)
- `-n numNodes`: Number of nodes to simulate (default: 1)

### Available Models

Run `./topo_expl` without arguments to see the list of available models. Each model represents a different hardware configuration:

## Example Usage: Print RCCL's algorithm/protocol selections

The tool is typically run with the `NCCL_DEBUG=INFO` environment variable, but for the convenience of just printing the algo/proto table, we use version `NCCL_DEBUG=version` in this example to avoid printing topo details.

```bash
# List available models
./topo_expl

# Test MI300 configuration (model 55)
NCCL_DEBUG=version ./topo_expl -m 55

# Test a multi-node MI300 configuration with 8 nodes
NCCL_DEBUG=version ./topo_expl -m 55 -n 8


# Test MI250 configuration (model 42)
NCCL_DEBUG=version ./topo_expl -m 42

# Test a multi-node MI250 configuration with 4 nodes
NCCL_DEBUG=version ./topo_expl -m 42 -n 4
```

