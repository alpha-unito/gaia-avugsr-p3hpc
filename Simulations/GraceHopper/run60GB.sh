#!/bin/bash 


cd $HOME_DIR/GraceHopper/test


for i in {1..3}; do
  sbatch test1GPUAdaptiveCPP.sh 60 100 $i
  sbatch test1GPUHIP.sh 60 100 $i
  sbatch test1GPUNVCPP.sh 60 100 $i
  sbatch test1GPUAdaptiveSycl.sh 60 100 $i
  sbatch test1GPUIntelSycl.sh 60 100 $i
  sbatch test1GPUNVOpenMP.sh 60 100 $i
  sbatch test1GPUCUDA.sh 60 100 $i
  sbatch test1GPULLVMOpenMP.sh 60 100 $i
done

