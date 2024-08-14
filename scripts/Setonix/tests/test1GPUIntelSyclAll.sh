#!/bin/bash --login
 
#SBATCH --account=pawsey0007-gpu
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#SBATCH --mem=0 
#SBATCH --time=04:00:00


module load rocm/5.7.3
module load gcc/12.2.0
module load craype-accel-amd-gfx90a


export COMP_PATH=/home/<user_name> /compiler/dpcpp

export GNUTOOLCHAIN=/opt/cray/pe/gcc/12.2.0/snos
export PATH=${COMP_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${COMP_PATH}/lib:${COMP_PATH}/boost-1.74.0/lib:$LD_LIBRARY_PATH
export CPATH=${COMP_PATH}/include:$CPATH

export GPUARCH=gfx90a
export HSA_XNACK=1 

export BOOST_HOME=/scratch/pawsey0007/<user_name>/src/compiler/boost_1_85_0/install
export LD_LIBRARY_PATH=$BOOST_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$BOOST_HOME/include:$CPATH



cd /scratch/pawsey0007/<user_name>/src/24_sc_gaia/Setonix/tests

##-----------------------------------------------------------------------------------------------------------------------------------------------
mkdir -p logs_test

export OMP_NUM_THREADS=1
# srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest  ../bin/GaiaGsrParSimIntelSycl.x -memGlobal $1 -IDtest 0 -itnlimit $2 > logs_test/log.1GPU_$1_isycl

for i in 1 2 3
do
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimIntelSycl.x -memGlobal 1 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_1_isycl_$i
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimIntelSycl.x -memGlobal 5 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_5_isycl_$i
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimIntelSycl.x -memGlobal 10 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_10_isycl_$i
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimIntelSycl.x -memGlobal 15 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_15_isycl_$i
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimIntelSycl.x -memGlobal 20 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_20_isycl_$i
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimIntelSycl.x -memGlobal 30 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_30_isycl_$i
    srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest ../bin/GaiaGsrParSimIntelSycl.x -memGlobal 60 -IDtest 0 -itnlimit 100 > logs_test/log.1GPU_60_isycl_$i
done