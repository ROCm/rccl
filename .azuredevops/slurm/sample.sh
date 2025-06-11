#!/bin/bash
#SBATCH --job-name=test_module-job
#SBATCH --output=test_module.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gt

source /etc/profile.d/lmod.sh
module load rocm/6.4.0
module list
which rocm-smi
