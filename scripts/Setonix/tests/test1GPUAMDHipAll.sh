#!/bin/bash --login
 
#SBATCH --account=pawsey0007-gpu
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#SBATCH --mem=0 
#SBATCH --time=02:00:00

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
export HSA_XNACK=0



cd /scratch/pawsey0007/<user_name>/src/24_sc_gaia/Setonix/tests

##-----------------------------------------------------------------------------------------------------------------------------------------------
mkdir -p logs_test

# export OMP_NUM_THREADS=1
# srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest  ../bin/GaiaGsrParSimHip.x -memGlobal $1 -IDtest 0 -itnlimit $2 > logs_test/log.1GPU_$1_hip


# export OMNITRACE_USE_ROCTRACER=1
# export OMNITRACE_USE_ROCPROF=1
# export OMNITRACE_ROCPROFILER_HSA_INTERCEPT=1
# export OMNITRACE_ROCPROFILER_API_TRACE=1
# export OMNITRACE_ROCPROFILER_KERNEL_TRACE=1
# export OMNITRACE_ROCPROFILER_MEM_TRACE=1

# export LD_DEBUG=libs

export OMP_NUM_THREADS=1
# srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimHip.x -memGlobal $1 -IDtest 0 -itnlimit $2 > logs_test/log.1GPU_$1_hip

for i in 1 2 3
do
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimHip.x -memGlobal 1 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_1_hip_$i
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimHip.x -memGlobal 5 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_5_hip_$i
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimHip.x -memGlobal 10 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_10_hip_$i
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimHip.x -memGlobal 15 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_15_hip_$i
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimHip.x -memGlobal 20 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_20_hip_$i
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimHip.x -memGlobal 30 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_30_hip_$i
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimHip.x -memGlobal 60 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_60_hip_$i
done
