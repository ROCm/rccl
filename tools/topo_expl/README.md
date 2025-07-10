# RCCL Topology Explorer (topo_expl)

The RCCL Topology Explorer is a tool for analyzing and exploring network topologies for RCCL (ROCm Communication Collectives Library) collective operations. It simulates various hardware configurations and predicts the performance of different collective communication algorithms and protocols.

## Overview

The topo_expl tool allows you to:
- Simulate different GPU cluster topologies using predefined XML models
- Analyze collective communication patterns (AllReduce, AllGather, Broadcast, etc.)
- Compare performance of different algorithms (Tree, Ring, CollNetDirect, CollNetChain)
- Evaluate protocol efficiency (LL, LL128, Simple)
- Determine optimal channel configurations for various data sizes

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

The Makefile will:
1. Create a `hipify_rccl` directory with transformed source files
2. Copy necessary headers and source files from the main RCCL codebase
3. Apply HIP transformations using hipify-perl
4. Compile the topo_expl executable

## Usage

```bash
./topo_expl -m model_id [-n numNodes=1]
```

### Parameters

- `-m model_id`: Specifies the topology model to use (required)
- `-n numNodes`: Number of nodes to simulate (default: 1)

### Available Models

Run `./topo_expl` without arguments to see the list of available models. Each model represents a different hardware configuration:

- **Rome-based configurations**: Various AMD EPYC Rome processor topologies
- **PCIe configurations**: Different PCIe interconnect patterns
- **CollNet configurations**: Collective network topologies
- **Multi-node setups**: Configurations for distributed systems

## Example Usage

```bash
# List available models
./topo_expl

# Test a single-node 8-GPU Rome configuration
./topo_expl -m 0

# Test a multi-node configuration with 4 nodes
./topo_expl -m 5 -n 4
```

## Output Format


Example output table:
```
| Max Size(B)     | Count           | Collective      | Algorithm  | Protocol   | Max Channels |
|-----------------|-----------------|-----------------|------------|------------|--------------|
| 1024            | 1000            | AllReduce       | Ring       | LL         | 8            |
| 65536           | 1000            | AllGather       | Tree       | Simple     | 16           |
```

## Model Files

Topology models are stored as XML files in the `models/` directory. Each model defines:
- GPU configurations and bus IDs
- Interconnect topology (PCIe, Infinity Fabric, etc.)
- Network interface configurations
- Bandwidth and latency characteristics

## License

This tool is part of the RCCL project and is subject to the same licensing terms as the main RCCL library.

