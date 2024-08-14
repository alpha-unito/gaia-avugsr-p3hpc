#!/bin/bash 

cd $HOME_DIR/CascadeLake/test

for i in {1..3}; do
  sbatch test1GPUAdaptiveCPP.sh 10 100 $i
  sbatch test1GPUHIP.sh 10 100 $i
  sbatch test1GPUNVCPP.sh 10 100 $i
  sbatch test1GPUAdaptiveSycl.sh 10 100 $i
  sbatch test1GPUIntelSycl.sh 10 100 $i
  sbatch test1GPUNVOpenMP.sh 10 100 $i
  sbatch test1GPUCUDA.sh 10 100 $i
  sbatch test1GPULLVMOpenMP.sh 10 100 $i
done
