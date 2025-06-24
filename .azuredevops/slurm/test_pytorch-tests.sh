#!/bin/bash
#SBATCH --job-name=pytorch-tests
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=300
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gt

short_id=$(hostname | cut -d'.' -f1 | cut -d'-' -f3-)
echo "Node identifier: $short_id"

source /etc/profile.d/lmod.sh
module load rocm/6.4.1
cd "$SLURM_SUBMIT_DIR"

docker run --rm --privileged --ipc=host --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined --pull always -v ${BINARIES_DIR}:/host rocm/pytorch:latest bash -c "/host/pytorch-tests/run_pytorch_tests.sh"
