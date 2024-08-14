#!/bin/bash 


cd $HOME_DIR/Epito/test



for i in {1..3}; do
  sbatch test1GPUAdaptiveCPP.sh 30 100 $i
  sbatch test1GPUHIP.sh 30 100 $i
  sbatch test1GPUNVCPP.sh 30 100 $i
  sbatch test1GPUAdaptiveSycl.sh 30 100 $i
  sbatch test1GPUIntelSycl.sh 30 100 $i
  sbatch test1GPUNVOpenMP.sh 30 100 $i
  sbatch test1GPUCUDA.sh 30 100 $i
  sbatch test1GPULLVMOpenMP.sh 30 100 $i
done

