#!/bin/bash --login
 
#SBATCH --account=pawsey0007-gpu
#SBATCH --partition=gpu-dev
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=32

#SBATCH --gpus-per-node=8
# #SBATCH --gpus-per-task=1

#SBATCH --mem=0 
#SBATCH --time=01:00:00

module load rocm/5.7.3
module load gcc/12.2.0
module load craype-accel-amd-gfx90a


export COMP_PATH=/home/<user_name> /compiler/AdaptivehipSYCL

export GNUTOOLCHAIN=/opt/cray/pe/gcc/12.2.0/snos
export PATH=${COMP_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${COMP_PATH}/lib:${COMP_PATH}/boost-1.74.0/lib:$LD_LIBRARY_PATH
export CPATH=${COMP_PATH}/include:$CPATH

export GPUARCH=gfx90a
export HSA_XNACK=1 


cd /scratch/pawsey0007/<user_name>/src/24_sc_gaia
cp Makefile.examples.AMD/Makefile_AMD_Acpp_CPP Makefile
make clean && make -j sycl
make -j syclgen
mv *.x ./Setonix/bin