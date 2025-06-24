#!/bin/bash
#SBATCH --job-name=rccl-tests
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=120
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gt

short_id=$(hostname | cut -d'.' -f1 | cut -d'-' -f3-)
echo "Node identifier: $short_id"

source /etc/profile.d/lmod.sh
module load rocm/6.4.1
cd "$BINARIES_DIR/bin"

export PATH="$BINARIES_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$BINARIES_DIR/lib:$LD_LIBRARY_PATH"

for coll in all_reduce all_gather reduce_scatter alltoall alltoallv broadcast gather reduce scatter sendrecv
do
	cmd="${MPI_HOME}/bin/mpirun -np 8 -mca oob_tcp_if_exclude docker,lo -mca btl_tcp_if_exclude docker,lo -mca pml ob1 -mca btl ^openib -x PATH -x LD_LIBRARY_PATH -x NCCL_DEBUG=VERSION -x NCCL_IGNORE_CPU_AFFINITY=1 -x HSA_NO_SCRATCH_RECLAIM=1 ${BINARIES_DIR}/bin/${coll}_perf -b 1K -e 1G -f 2 -g 1 -d float -n 100 -w 50 -Z json -x rccl-tests_${coll}_nodes1_gpus8_float.json"

	echo "Running ${coll}"
	echo "Run cmd: ${cmd}"
	eval ${cmd}

	sleep 2
done

## To add
### Summarize results
### Convert to junit
