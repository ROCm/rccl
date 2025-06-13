#!/bin/bash
#SBATCH --job-name=rccl-test
#SBATCH --output=rccl-test-%j.out
#SBATCH --error=rccl-test-%j.out
#SBATCH --time=120
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gt

short_id=$(hostname | cut -d'-' -f3-)
echo "Node identifier: $short_id"

source /etc/profile.d/lmod.sh
module load rocm/6.4.0
cd "$BINARIES_DIR/bin"
./rccl-UnitTests --gtest_output=xml:$PIPELINE_WORKSPACE/test_output.xml --gtest_color=yes
