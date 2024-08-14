#!/bin/bash
#SBATCH --job-name=Gaia_asycl
#SBATCH -p cascadelake

##SBATCH --nodelist=epito01

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
# Number of GPUs per node
# #SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1

# #SBATCH -o out1Node.log
# #SBATCH -e out1Node.err
#SBATCH -t 06:00:00

source ~/gcc_cascadelake
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

export COMP_PATH=/home/<user_name>/src/compiler/CASCADELAKE

export GNUTOOLCHAIN=/home/<user_name>/src/compiler/CASCADELAKE/gcc-12.2.0
export PATH=${COMP_PATH}/AdaptiveCPP_new/bin:$PATH
export LD_LIBRARY_PATH=${COMP_PATH}/AdaptiveCPP_new/lib:${COMP_PATH}/boost-1.74.0/lib:$LD_LIBRARY_PATH
export CPATH=${COMP_PATH}/AdaptiveCPP_new/include:$CPATH
export MPI_HOME=/opt/openmpi

export GPUARCH=sm_75
source /home/<user_name>/src/compiler/CASCADELAKE/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/hpcx/hpcx-2.17.1/hpcx-init.sh
hpcx_load
export MPI_HOME=/home/<user_name>/src/compiler/CASCADELAKE/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi
export CUDA_HOME=/home/<user_name>/src/compiler/CASCADELAKE/hpc_sdk/Linux_x86_64/24.3/cuda

##-----------------------------------------------------------------------------------------------------------------------------------------------
cd $HOME_DIR 
cp Makefile.examples/Makefile_NVIDIA_Acpp_Sycl Makefile 
make clean && make -j sycl
mv *.x ./TeslaT4/bin

