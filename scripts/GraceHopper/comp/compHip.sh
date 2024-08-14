#!/bin/bash
#SBATCH --job-name=Gaia_omp
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
export HIP_PLATFORM=nvidia
export HIP_HOME=/home/<user_name>/src/compiler/GRACEHOPPER/rocm-5.7/clr/build/install

export GNUTOOLCHAIN=/home/<user_name>/src/compiler/GRACEHOPPER/gcc-12.2.0
export PATH=${HIP_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${HIP_HOME}/lib:$LD_LIBRARY_PATH
export CPATH=${HIP_HOME}/include:$CPATH

export GPUARCH=sm_90
source /home/<user_name>/src/compiler/GRACEHOPPER/hpc_sdk/Linux_aarch64/24.3/comm_libs/12.3/hpcx/hpcx-2.17.1/hpcx-init.sh
hpcx_load
export MPI_HOME=/home/<user_name>/src/compiler/GRACEHOPPER/hpc_sdk/Linux_aarch64/24.3/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi
export CUDA_HOME=/home/<user_name>/src/compiler/GRACEHOPPER/hpc_sdk/Linux_aarch64/24.3/cuda


##-----------------------------------------------------------------------------------------------------------------------------------------------
cd $HOME_DIR
make clean && make -j hip
mv *.x ./GraceHopper/bin

