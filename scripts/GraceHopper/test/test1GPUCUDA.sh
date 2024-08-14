#!/bin/bash
#SBATCH --job-name=Gaia_cuda
#SBATCH -p gracehopper

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
# Number of GPUs per node
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1

# #SBATCH -o out1Node.log
# #SBATCH -e out1Node.err
#SBATCH -t 06:00:00

source ~/gcc_gracehopper.sh
export COMP_PATH=/home/<user_name>/src/compiler/GRACEHOPPER/hpc_sdk

export GNUTOOLCHAIN=/home/<user_name>/src/compiler/GRACEHOPPER/gcc-12.2.0
export PATH=${COMP_PATH}/Linux_aarch64/24.3/compilers/bin:$PATH

export GPUARCH=sm_90
source /home/<user_name>/src/compiler/GRACEHOPPER/hpc_sdk/Linux_aarch64/24.3/comm_libs/12.3/hpcx/hpcx-2.17.1/hpcx-init.sh
hpcx_load
export MPI_HOME=/home/<user_name>/src/compiler/GRACEHOPPER/hpc_sdk/Linux_aarch64/24.3/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi
export CUDA_HOME=/home/<user_name>/src/compiler/GRACEHOPPER/hpc_sdk/Linux_aarch64/24.3/cuda



##-----------------------------------------------------------------------------------------------------------------------------------------------
mkdir -p logs_test
mpirun -np 1 --mca pml ucx --mca btl_tcp_if_include mlx5_1 ../bin/GaiaGsrParSimCuda.x -memGlobal $1 -IDtest 0 -itnlimit $2 > logs_test/log.1GPU_$1_$2itn_cuda_$3


