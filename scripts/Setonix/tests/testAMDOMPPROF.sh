#!/bin/bash --login
 
#SBATCH --account=pawsey0007-gpu
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#SBATCH --mem=0 
#SBATCH --time=01:00:00


module load omnitrace
module load rocm/5.7.3
module load gcc/12.2.0
module load craype-accel-amd-gfx90a


export COMP_PATH=/home/<user_name> /compiler/AdaptivehipSYCL

export GNUTOOLCHAIN=/opt/cray/pe/gcc/12.2.0/snos
export PATH=${COMP_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${COMP_PATH}/lib:${COMP_PATH}/boost-1.74.0/lib:$LD_LIBRARY_PATH
export CPATH=${COMP_PATH}/include:$CPATH

export GPUARCH=gfx90a

export BOOST_HOME=/scratch/pawsey0007/<user_name>/src/compiler/boost_1_85_0/install
export LD_LIBRARY_PATH=$BOOST_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$BOOST_HOME/include:$CPATH
export HSA_XNACK=1 

cd /scratch/pawsey0007/<user_name>/src/24_sc_gaia/Setonix/tests

##-----------------------------------------------------------------------------------------------------------------------------------------------
mkdir -p logs_test


export OMNITRACE_USE_ROCTRACER=1
export OMNITRACE_USE_ROCPROF=1
export OMNITRACE_ROCPROFILER_HSA_INTERCEPT=1
export OMNITRACE_ROCPROFILER_API_TRACE=1
export OMNITRACE_ROCPROFILER_KERNEL_TRACE=1
export OMNITRACE_ROCPROFILER_MEM_TRACE=1
export LD_DEBUG=libs

export OMP_NUM_THREADS=1
# srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest rocprof --stats --hsa-trace --hip-trace --list-basic ../bin/GaiaGsrParSimOMPGpu.x -memGlobal $1 -IDtest 0 -itnlimit $2 > logs_test/log.1GPU_$1_omp
srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest rocprof --stats --hsa-trace --hip-trace ../bin/GaiaGsrParSimOMPGpu.x -memGlobal $1 -IDtest 0 -itnlimit $2 > prof_test/log.1GPU_$1_omp
