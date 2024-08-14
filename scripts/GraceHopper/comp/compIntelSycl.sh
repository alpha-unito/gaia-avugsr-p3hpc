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
export COMP_PATH=/home/<user_name>/src/compiler/GRACEHOPPER

#export GNUTOOLCHAIN=/home/<user_name>/spack/opt/spack/linux-ubuntu22.04-neoverse_v2/gcc-11.4.0/gcc-11.4.0-2uegve2ficupltcugt452iiad7m7hvo4
export GNUTOOLCHAIN=/home/<user_name>/src/compiler/GRACEHOPPER/gcc-12.2.0


export PATH=${COMP_PATH}/IntelSycl/bin:$PATH
export LD_LIBRARY_PATH=${COMP_PATH}/IntelSycl/lib:$LD_LIBRARY_PATH
export CPATH=${COMP_PATH}/IntelSycl/include:$CPATH

export GPUARCH=sm_90
source /home/<user_name>/src/compiler/GRACEHOPPER/hpc_sdk/Linux_aarch64/24.3/comm_libs/12.3/hpcx/hpcx-2.17.1/hpcx-init.sh
hpcx_load
export MPI_HOME=/home/<user_name>/src/compiler/GRACEHOPPER/hpc_sdk/Linux_aarch64/24.3/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi
export CUDA_HOME=/home/<user_name>/src/compiler/GRACEHOPPER/hpc_sdk/Linux_aarch64/24.3/cuda

export CUDA_VISIBLE_DEVICES=0


##-----------------------------------------------------------------------------------------------------------------------------------------------
cd $HOME_DIR 
cp Makefile.examples/Makefile_NVIDIA_Intel_Sycl Makefile 
make clean && make -j sycl
mv *.x ./GraceHopper/bin

