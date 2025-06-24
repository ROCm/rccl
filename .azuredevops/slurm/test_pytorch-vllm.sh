#!/bin/bash
#SBATCH --job-name=pytorch-vllm
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
cd "$SLURM_SUBMIT_DIR"

docker run --rm --privileged --ipc=host --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined --pull always -v ${BINARIES_DIR}:/host -v /mnt/GT_WEKA_NFS/new_gt/models:/data rocm/vllm-dev:nightly bash -c "ls /host && ls /host/lib && ls /host/pytorch-vllm && /host/pytorch-vllm/run_pytorch_vllm.sh"
