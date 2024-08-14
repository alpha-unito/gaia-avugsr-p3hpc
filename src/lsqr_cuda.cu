/* lsqr.c
   This C version of LSQR was first created by
      Michael P Friedlander <mpf@cs.ubc.ca>
   as part of his BCLS package:
      http://www.cs.ubc.ca/~mpf/bcls/index.html.
   The present file is maintained by
      Michael Saunders <saunders@stanford.edu>

   31 Aug 2007: First version of this file lsqr.c obtained from
                Michael Friedlander's BCLS package, svn version number
                $Revision: 273 $ $Date: 2006-09-04 15:59:04 -0700 (Mon, 04 Sep 2006) $

                The stopping rules were slightly altered in that version.
                They have been restored to the original rules used in the f77 LSQR.

Parallelized for ESA Gaia Mission. U. becciani A. Vecchiato 2013
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include "util.h"
#include <limits.h>
#include <sys/time.h>
#include "lsqr.h"

#include <iostream>

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}



#define ZERO   0.0
#define ONE    1.0

// static const int blockSize = 256;
// static const int gridSize = 4096;
static const int blockSize = 128;
static const int gridSize = 1024;


#if defined(__NVIDIA90__)
    #define ThreadsXBlock 64
    #define BlockXGrid 1024
    #define ThreadsXBlockAprod2Astro 32
    #define BlockXGridAprod2Astro 1024
    #define TILE_WIDTH 32 
#elif defined (__NVIDIA80__)
    #define ThreadsXBlock 	128
    #define BlockXGrid 	1024
    #define ThreadsXBlockAprod2Astro  64	
    #define BlockXGridAprod2Astro  1024
    #define TILE_WIDTH  1024
#elif defined(__NVIDIA70__)
    #define ThreadsXBlock 	32
    #define BlockXGrid 	128
    #define ThreadsXBlockAprod2Astro  32	
    #define BlockXGridAprod2Astro  128
    #define TILE_WIDTH  32
#else
    #error "Unknown platform"
#endif



__global__ void dknorm_compute(double* dknorm_vec,
                                const double* wVect_dev,
                                const long begin, 
                                const long end,
                                const double t3){
    auto gthIdx = blockIdx.x * blockDim.x + threadIdx.x+begin;
    const auto gridSize = blockSize*gridDim.x;
    __shared__ double shArr[blockSize];
    shArr[threadIdx.x] = 0.0;
    *dknorm_vec=0.0;
    for (auto i = gthIdx; i < end; i += gridSize){
        shArr[threadIdx.x] += wVect_dev[i]*wVect_dev[i]*t3*t3;
    }

    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { 
        if (threadIdx.x<size)
            shArr[threadIdx.x] += shArr[threadIdx.x+size];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(dknorm_vec,shArr[0]);
}

template<typename T>
__global__ void maxCommMultiBlock_double (double *gArr, double *gOut, const T arraySize) {
    const T gthIdx = threadIdx.x + blockIdx.x*blockSize;
    const T gridSize = blockSize*gridDim.x;
    __shared__ double shArr[blockSize];
    shArr[threadIdx.x] = fabs(gArr[0]);
    for (T i = gthIdx; i < arraySize; i += gridSize){
        if(shArr[threadIdx.x] < fabs(gArr[i])) shArr[threadIdx.x] = fabs(gArr[i]);
    }

    __syncthreads();
    for (int size = blockSize/2; size > 0; size /= 2) { 
        if (threadIdx.x < size)
            if(shArr[threadIdx.x] < shArr[threadIdx.x + size]) shArr[threadIdx.x] = shArr[threadIdx.x + size];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        gOut[blockIdx.x] = shArr[0];
 }

template<typename T>
__global__ void sumCommMultiBlock_double(double *gArr, double *gOut, const double max, const T arraySize) {
    
    const T gthIdx = threadIdx.x + blockIdx.x*blockSize;
    const T gridSize = blockSize*gridDim.x;
    __shared__ double shArr[blockSize];
    shArr[threadIdx.x] = 0.0;
    double divmax=1/max;
    for (T i = gthIdx; i < arraySize; i += gridSize)
        shArr[threadIdx.x] += (gArr[i]*divmax)*(gArr[i]*divmax);

    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { 
        if (threadIdx.x<size)
            shArr[threadIdx.x] += shArr[threadIdx.x+size];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        gOut[blockIdx.x] = shArr[0];
}

template<typename T>
__global__ void realsumCommMultiBlock_double(double *gArr, double *gOut, const T arraySize) {
    
    __shared__ double shArr[blockSize];
    T gthIdx = threadIdx.x + blockIdx.x*blockSize;
    const T gridSize = blockSize*gridDim.x;
    shArr[threadIdx.x] = 0.0;

    for (T i = gthIdx; i < arraySize; i += gridSize)
        shArr[threadIdx.x] += gArr[i];

    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { 
        if (threadIdx.x<size)
            shArr[threadIdx.x] += shArr[threadIdx.x+size];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        gOut[blockIdx.x] = shArr[0];
}


template<typename T>
__global__ void dscal(double* __restrict__ knownTerms_dev, const double val, const T N, const double sign)
{
    T ix = blockIdx.x * blockDim.x + threadIdx.x;

    while(ix < N){

        knownTerms_dev[ix]=sign*(knownTerms_dev[ix]*val);

        ix+=gridDim.x*blockDim.x;

    }
}

__global__ void  kAuxcopy_Kernel(double* __restrict__ knownTerms_dev, double* __restrict__ kAuxcopy_dev, const long nobs, const int N)
{
    const long ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ix < N) {
        kAuxcopy_dev[ix] = knownTerms_dev[nobs + ix];
        knownTerms_dev[nobs + ix] = 0.0;
    }
}


__global__ void vAuxVect_Kernel(double* __restrict__ vVect_dev, double* __restrict__ vAuxVect_dev, const long N)
{
    const long ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ix < N) {
        vAuxVect_dev[ix] = vVect_dev[ix];
        vVect_dev[ix] = 0.0;
    }
}

__global__ void vVect_Put_To_Zero_Kernel(double* __restrict__ vVect_dev, const long localAstroMax, const long nunkSplit)
{
    const long ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ix >= localAstroMax && ix < nunkSplit) {
        vVect_dev[ix] = 0.0;
    }
}


__global__ void kauxsum (double* __restrict__ knownTerms_dev,const double* __restrict__ kAuxcopy_dev, const int n)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ix < n) {
        knownTerms_dev[ix] = knownTerms_dev[ix]+kAuxcopy_dev[ix];
    }
}


__global__ void vaux_sum(double* __restrict__  vV, const double* __restrict__  vA, const long lAM)
{
    const long ix = blockIdx.x * blockDim.x + threadIdx.x;

    if(ix<lAM){
        vV[ix]+=vA[ix];
    }
} 

__global__ void transform1(double* __restrict__  xSolution, const double* __restrict__ wVect, const long begin, const long end, const double t1){
    long ix = blockIdx.x * blockDim.x + threadIdx.x+begin;

    while(ix < end){

        xSolution[ix]   =  xSolution[ix] + t1*wVect[ix];

        ix+=gridDim.x*blockDim.x;
    }
}

__global__ void transform2(double* __restrict__  standardError, const double* __restrict__  wVect, const long begin, const long end, const double t3){
    long ix = blockIdx.x * blockDim.x + threadIdx.x+begin;

    while(ix < end){
        standardError[ix]  =  standardError[ix] +(t3*wVect[ix])*(t3*wVect[ix]);

        ix+=gridDim.x*blockDim.x;
    }
}

__global__ void transform3(double* __restrict__ wVect, const double* __restrict__ vVect, const long begin, const long end, const double t2){
    long ix = blockIdx.x * blockDim.x + threadIdx.x+begin;

    while(ix < end){
        wVect[ix]   =  vVect[ix]+t2*wVect[ix];

        ix+=gridDim.x*blockDim.x;
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


// Astrometric part
__global__ void aprod1_Kernel_astro(double* __restrict__ knownTerms_dev,
                                    const double* __restrict__ systemMatrix_dev,
                                    const double* __restrict__ vVect_dev, 
                                    const long* __restrict__ matrixIndexAstro_dev, 
                                    const long mapNoss, 
                                    const long offLocalAstro, 
                                    const short  nAstroPSolved){
    double sum = 0.0;
    long jstartAstro = 0;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (ix < mapNoss) {
        sum = 0.0;
        jstartAstro = matrixIndexAstro_dev[ix] - offLocalAstro;
        #pragma unroll
        for(short jx = 0; jx < nAstroPSolved; jx++) {
           sum = sum + systemMatrix_dev[ix*nAstroPSolved + jx] * vVect_dev[jstartAstro+jx];
        }
        knownTerms_dev[ix] = knownTerms_dev[ix] + sum;
        ix+=gridDim.x*blockDim.x;
    }
}


// Attitude part
__global__ void aprod1_Kernel_att_AttAxis(double* __restrict__ knownTerms_dev, 
                                        const double* __restrict__ systemMatrix_dev, 
                                        const double* __restrict__ vVect_dev, 
                                        const long* __restrict__ matrixIndexAtt_dev, 
                                        const long  nAttP, 
                                        const long mapNoss, 
                                        const long nDegFreedomAtt, 
                                        const long offLocalAtt, 
                                        const short nAttParAxis)
{
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    while (ix < mapNoss) {
        double sum = 0.0;
        const long miValAtt=matrixIndexAtt_dev[ix];
        long jstartAtt_0 = miValAtt + offLocalAtt; 
        #pragma unroll
        for(short inpax = 0;inpax<nAttParAxis;++inpax)
            sum = sum + systemMatrix_dev[ix * nAttP + inpax] * vVect_dev[jstartAtt_0 + inpax];
        jstartAtt_0 += nDegFreedomAtt;
        #pragma unroll
        for(short inpax = 0;inpax<nAttParAxis;inpax++)
            sum = sum + systemMatrix_dev[ix * nAttP + nAttParAxis + inpax ] * vVect_dev[jstartAtt_0 + inpax];
        jstartAtt_0 += nDegFreedomAtt;
        #pragma unroll
        for(short inpax = 0;inpax<nAttParAxis;inpax++)
            sum = sum + systemMatrix_dev[ix * nAttP + nAttParAxis + nAttParAxis + inpax] * vVect_dev[jstartAtt_0 + inpax];

        knownTerms_dev[ix] = knownTerms_dev[ix] + sum;
        ix+=gridDim.x*blockDim.x;
  }
}



// Instrumental part
__global__ void aprod1_Kernel_instr(double* __restrict__ knownTerms_dev, 
                                    const double* __restrict__ systemMatrix_dev, 
                                    const double* __restrict__ vVect_dev, 
                                    const int* __restrict__ instrCol_dev, 
                                    const long mapNoss, 
                                    const long offLocalInstr, 
                                    const short  nInstrPSolved)
{
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    while (ix < mapNoss) {
        double sum = 0.0;
        const long iiVal=ix*nInstrPSolved;
        long ixInstr = 0;
        for(short inInstr=0;inInstr<nInstrPSolved;inInstr++){
            ixInstr=offLocalInstr+instrCol_dev[iiVal+inInstr];
            sum += systemMatrix_dev[ix * nInstrPSolved + inInstr]*vVect_dev[ixInstr];
        }
        knownTerms_dev[ix] = knownTerms_dev[ix] + sum;
        ix+=gridDim.x*blockDim.x;
    }
}



// Global part
__global__ void aprod1_Kernel_glob(double* __restrict__ knownTerms_dev, 
                                    const double* __restrict__ systemMatrix_dev, 
                                    const double* __restrict__ vVect_dev, 
                                    const long offLocalGlob, 
                                    const long mapNoss, 
                                    const short nGlobP)
{
    double sum = 0.0;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;   
    while (ix < mapNoss) {
        sum = 0.0;
        for(short inGlob=0;inGlob<nGlobP;inGlob++){
            sum=sum+systemMatrix_dev[ix * nGlobP + inGlob]*vVect_dev[offLocalGlob+inGlob];
        }
        knownTerms_dev[ix] = knownTerms_dev[ix] + sum;
        ix+=gridDim.x*blockDim.x;
    }
}


// //  CONSTRAINTS OF APROD MODE 1
/// ExtConstr
/// Mode 1 ExtConstr
__global__ void aprod1_Kernel_ExtConstr(double* __restrict__ knownTerms_dev,
                                        const double* __restrict__ systemMatrix_dev, 
                                        const double* __restrict__ vVect_dev, 
                                        const long VrIdAstroPDimMax,
                                        const long mapNoss, 
                                        const long nDegFreedomAtt, 
                                        const int startingAttColExtConstr, 
                                        const int nEqExtConstr, 
                                        const int nOfElextObs, 
                                        const int numOfExtStar, 
                                        const int numOfExtAttCol, 
                                        const short nAstroPSolved, 
                                        const short nAttAxes)
{
    long offExtAtt;
    long offExtAttConstr = VrIdAstroPDimMax*nAstroPSolved+startingAttColExtConstr;
    long vVIx;
    long ktIx = mapNoss;
    long offExtConstr;
    long j3 = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;

    for (long iexc = 0; iexc < nEqExtConstr; iexc++) {
        sum=0.0;
        offExtConstr = iexc*nOfElextObs;
        if (j3 < numOfExtStar*nAstroPSolved)
            sum = sum + systemMatrix_dev[offExtConstr+j3]*vVect_dev[j3];
        for (short nax = 0; nax < nAttAxes; nax++) {
            offExtAtt = offExtConstr + numOfExtStar*nAstroPSolved + nax*numOfExtAttCol;
            vVIx=offExtAttConstr+nax*nDegFreedomAtt;
            if (j3 < numOfExtAttCol) sum += systemMatrix_dev[offExtAtt+j3]*vVect_dev[vVIx+j3];
        }

        atomicAdd(&knownTerms_dev[ktIx+iexc], sum);
    }
}




/// BarConstr
/// Mode 1 BarConstr
__global__ void aprod1_Kernel_BarConstr(double* __restrict__ knownTerms_dev,
                                        const double* __restrict__ systemMatrix_dev, 
                                        const double* __restrict__ vVect_dev,
                                        const int nOfElextObs, 
                                        const int nOfElBarObs, 
                                        const int nEqExtConstr, 
                                        const int mapNoss, 
                                        const int nEqBarConstr, 
                                        const int numOfBarStar, 
                                        const short nAstroPSolved){
    long offBarConstrIx;
    long ktIx = mapNoss + nEqExtConstr;    
    long j3 = blockIdx.x * blockDim.x + threadIdx.x;   

    for(int iexc=0;iexc<nEqBarConstr;iexc++ ){
        double sum=0.0;
        offBarConstrIx=iexc*nOfElBarObs;
        if (j3 < numOfBarStar*nAstroPSolved)
            sum = sum + systemMatrix_dev[offBarConstrIx+j3]*vVect_dev[j3];

        atomicAdd(&knownTerms_dev[ktIx+iexc],sum);
    }
}



/// InstrConstr
/// Mode 1 InstrConstr
__global__ void aprod1_Kernel_InstrConstr(double* __restrict__ knownTerms_dev,
                                        const double* __restrict__ systemMatrix_dev, 
                                        const double* __restrict__ vVect_dev,
                                        const int* __restrict__ instrConstrIlung_dev, 
                                        const int* __restrict__ instrCol_dev, 
                                        const long VrIdAstroPDimMax, 
                                        const long mapNoss, 
                                        const long nDegFreedomAtt, 
                                        const int nOfElextObs, 
                                        const int nEqExtConstr, 
                                        const int nOfElBarObs, 
                                        const int nEqBarConstr, 
                                        const int myid, 
                                        const int nOfInstrConstr, 
                                        const int nproc, 
                                        const short nAstroPSolved, 
                                        const short nAttAxes, 
                                        const short nInstrPSolved){

    const long ktIx=mapNoss+nEqExtConstr+nEqBarConstr;    
    const short i1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int i1_Aux = myid + i1*nproc;

    long offSetInstrConstr1=VrIdAstroPDimMax*nAstroPSolved+nDegFreedomAtt*nAttAxes;
    long offSetInstrInc=nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr;
    long offvV=0;

    int offSetInstr=0;
    int vVix=0;
    
    double sum = 0.0;

    
    if(i1_Aux < nOfInstrConstr){
        sum=0.0;
        offSetInstr=0;
        for(int m=0;m<i1_Aux;m++)
        {
            offSetInstrInc+=instrConstrIlung_dev[m];
            offSetInstr+=instrConstrIlung_dev[m];
        }
        offvV=mapNoss*nInstrPSolved+offSetInstr;
        for(int j3 = 0; j3 < instrConstrIlung_dev[i1_Aux]; j3++)
        {
            vVix=instrCol_dev[offvV+j3];
            sum=sum+systemMatrix_dev[offSetInstrInc+j3]*vVect_dev[offSetInstrConstr1+vVix];
        }
        atomicAdd(&knownTerms_dev[ktIx+i1_Aux],sum);
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


__global__ void aprod2_Kernel_astro(double * __restrict__ vVect_dev, 
                                    const double * __restrict__ systemMatrix_dev, 
                                    const double * __restrict__ knownTerms_dev, 
                                    const long* __restrict__ matrixIndexAstro_dev, 
                                    const long* __restrict__ startend_dev,  
                                    const long offLocalAstro, 
                                    const long mapNoss, 
                                    const short nAstroPSolved){   
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    while(ix < mapNoss)
    {
        long tid=matrixIndexAstro_dev[startend_dev[ix]];
        for(long i=startend_dev[ix]; i<startend_dev[ix+1]; ++i){
            #pragma unroll
            for (short jx = 0; jx < nAstroPSolved; ++jx)
                vVect_dev[tid - offLocalAstro + jx]+= systemMatrix_dev[i*nAstroPSolved + jx] * knownTerms_dev[i];
        }
        ix+=gridDim.x*blockDim.x;
   }
}


__global__ void aprod2_Kernel_att_AttAxis(double * __restrict__ vVect_dev, 
                                            const double * __restrict__ systemMatrix_dev, 
                                            const double * __restrict__ knownTerms_dev, 
                                            const long* __restrict__ matrixIndexAtt_dev, 
                                            const long nAttP, 
                                            const long nDegFreedomAtt, 
                                            const long offLocalAtt, 
                                            const long mapNoss, 
                                            const short nAstroPSolved, 
                                            const short nAttParAxis){
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    while(ix < mapNoss)
    {
        long jstartAtt = matrixIndexAtt_dev[ix] + offLocalAtt;
        #pragma unroll
        for (short inpax = 0; inpax < nAttParAxis; inpax++)
            atomicAdd(&vVect_dev[jstartAtt + inpax],systemMatrix_dev[ix * nAttP + inpax] * knownTerms_dev[ix]);
        jstartAtt +=nDegFreedomAtt;
        #pragma unroll
        for (short inpax = 0; inpax < nAttParAxis; inpax++)
            atomicAdd(&vVect_dev[jstartAtt + inpax],systemMatrix_dev[ix * nAttP + nAttParAxis + inpax] * knownTerms_dev[ix]);
        jstartAtt +=nDegFreedomAtt;
        #pragma unroll
        for (short inpax = 0; inpax < nAttParAxis; inpax++)
            atomicAdd(&vVect_dev[jstartAtt + inpax],systemMatrix_dev[ix * nAttP + nAttParAxis+nAttParAxis + inpax] * knownTerms_dev[ix]);
        ix+=gridDim.x*blockDim.x;
   }
}




__global__ void aprod2_Kernel_instr(double * __restrict__ vVect_dev, 
                                    const double * __restrict__ systemMatrix_dev, 
                                    const double * __restrict__ knownTerms_dev, 
                                    const int * __restrict__ instrCol_dev,  
                                    const long offLocalInstr, 
                                    const long mapNoss, 
                                    const short nInstrPSolved){
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    while(ix < mapNoss)
    {
        for (short inInstr = 0; inInstr < nInstrPSolved; inInstr++)
            atomicAdd(&vVect_dev[offLocalInstr + instrCol_dev[ix*nInstrPSolved+inInstr]],systemMatrix_dev[ix*nInstrPSolved + inInstr] * knownTerms_dev[ix]);
        ix+=gridDim.x*blockDim.x;
   }
}




__global__ void sumCommMultiBlock_double_aprod2_Kernel_glob(double * __restrict__ dev_vVect_glob_sum, 
                                                    const double * __restrict__ systemMatrix_dev, 
                                                    const double * __restrict__ knownTerms_dev, 
                                                    const double * __restrict__ vVect_dev, 
                                                    const long nGlobP, 
                                                    const long mapNoss,  
                                                    const long offLocalGlob, 
                                                    const int inGlob)
{
    
    long gthIdx = threadIdx.x + blockIdx.x*blockSize;
    const int gridSize = blockSize*gridDim.x;
    __shared__ double shArr[blockSize];
    shArr[threadIdx.x] = 0.0;
    for (long ix = gthIdx; ix < mapNoss; ix += gridSize)
        shArr[threadIdx.x] += systemMatrix_dev[ix * nGlobP + inGlob] * knownTerms_dev[ix];
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (threadIdx.x<size)
            shArr[threadIdx.x] += shArr[threadIdx.x+size];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        dev_vVect_glob_sum[blockIdx.x] = shArr[0];
}



__global__ void realsumCommMultiBlock_double_aprod2_Kernel_glob(double * __restrict__ vVect_dev, 
                                                            const double * __restrict__ gArr, 
                                                            const long arraySize, 
                                                            const long offLocalGlob, 
                                                            const short inGlob)
{
    int thIdx = threadIdx.x;
    long gthIdx = thIdx + blockIdx.x*blockSize;
    const int gridSize = blockSize*gridDim.x;
    double sum = 0.0;
    for (long i = gthIdx; i < arraySize; i += gridSize)
        sum += gArr[i];
    __shared__ double shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (thIdx<size)
            shArr[thIdx] += shArr[thIdx+size];
        __syncthreads();
    }
    if (thIdx == 0)
    {
        vVect_dev[offLocalGlob + inGlob] = vVect_dev[offLocalGlob + inGlob] + shArr[0];
    }
}


                                        

//  CONSTRAINTS OF APROD MODE 2
__global__ void aprod2_Kernel_ExtConstr(double* __restrict__ vVect_dev, 
                                        const double* __restrict__ systemMatrix_dev, 
                                        const double* __restrict__ knownTerms_dev, 
                                        const long mapNoss, 
                                        const long nDegFreedomAtt, 
                                        const long VrIdAstroPDimMax, 
                                        const int nEqExtConstr, 
                                        const int nOfElextObs, 
                                        const int numOfExtStar, 
                                        const int startingAttColExtConstr, 
                                        const int numOfExtAttCol,
                                        const short  nAttAxes, 
                                        const short nAstroPSolved){
    const long off1 = VrIdAstroPDimMax*nAstroPSolved+startingAttColExtConstr;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    long off2;
    long off3;
    long offExtAttConstrEq;
    long offExtUnk;

    double yi;

    for(int ix = 0; ix < nEqExtConstr; ix++ ){  //stars
        yi = knownTerms_dev[mapNoss + ix];
        if (i < numOfExtStar) {
            off3 = i*nAstroPSolved;
            off2 = ix*nOfElextObs + off3;
            for(short j2 = 0; j2 < nAstroPSolved; j2++){
                vVect_dev[j2+off3] += systemMatrix_dev[off2+j2]*yi;
            }
        }
    } 

    for(int ix=0;ix<nEqExtConstr;ix++ ){  //att
        yi = knownTerms_dev[mapNoss + ix];
        offExtAttConstrEq =  ix*nOfElextObs + numOfExtStar*nAstroPSolved; //Att offset inside ix row
        for(short nax = 0; nax < nAttAxes; nax++){
            offExtUnk = off1 + nax*nDegFreedomAtt;// Att offset for Unk array on extConstr
            off2=offExtAttConstrEq+nax*numOfExtAttCol;

            if (i < numOfExtAttCol) {
                vVect_dev[offExtUnk+i] = vVect_dev[offExtUnk+i] + systemMatrix_dev[off2+i]*yi;
            }
        }
    }
}



__global__ void aprod2_Kernel_BarConstr (double* __restrict__ vVect_dev,
                                        const double* __restrict__ systemMatrix_dev, 
                                        const double* __restrict__ knownTerms_dev, 
                                        const long mapNoss, 
                                        const int nEqBarConstr, 
                                        const int nEqExtConstr, 
                                        const int nOfElextObs, 
                                        const int nOfElBarObs, 
                                        const int numOfBarStar, 
                                        const short nAstroPSolved){    

    const int yx = blockIdx.x * blockDim.x + threadIdx.x;
    double yi;
    long offBarStarConstrEq;
    
    for(int ix=0;ix<nEqBarConstr;ix++ ){  //stars
        yi = knownTerms_dev[mapNoss+nEqExtConstr+ix];
        offBarStarConstrEq = nEqExtConstr*nOfElextObs+ix*nOfElBarObs;//Star offset on the row i2 (all the row)
        if (yx < numOfBarStar) {
            for(short j2=0;j2<nAstroPSolved;j2++)
                vVect_dev[j2+yx*nAstroPSolved] = vVect_dev[j2+yx*nAstroPSolved] + systemMatrix_dev[offBarStarConstrEq+yx*nAstroPSolved+j2]*yi;
        }
    } 
}




__global__ void aprod2_Kernel_InstrConstr(double* __restrict__ vVect_dev,
                                        const double* __restrict__ systemMatrix_dev, 
                                        const double* __restrict__ knownTerms_dev, 
                                        const int* __restrict__ instrConstrIlung_dev, 
                                        const int* __restrict__ instrCol_dev, 
                                        const long VrIdAstroPDimMax, 
                                        const long nDegFreedomAtt, 
                                        const long mapNoss, 
                                        const int nEqExtConstr, 
                                        const int nEqBarConstr, 
                                        const int nOfElextObs, 
                                        const int nOfElBarObs, 
                                        const int myid, 
                                        const int nOfInstrConstr, 
                                        const int nproc, 
                                        const short nAstroPSolved, 
                                        const short  nAttAxes, 
                                        const short  nInstrPSolved)
{
    const long k1 = blockIdx.x * blockDim.x + threadIdx.x;
    long k1_Aux = myid + k1*nproc;
    const long off3=nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr;
    const long offInstrUnk=VrIdAstroPDimMax*nAstroPSolved+nAttAxes*nDegFreedomAtt;
    const long off2=mapNoss+nEqExtConstr+nEqBarConstr;
    const long off4=mapNoss*nInstrPSolved;
    double yi;
    
    if(k1_Aux < nOfInstrConstr) {
        yi=knownTerms_dev[off2+k1_Aux];
        int offSetInstr=0;
        for(long m=0;m<k1_Aux;m++)
            offSetInstr+=instrConstrIlung_dev[m];

        const long off1=off3+offSetInstr;
        const long off5=off4+offSetInstr;
        for(int j = 0; j < instrConstrIlung_dev[k1_Aux]; j++) {
                atomicAdd(&vVect_dev[offInstrUnk+instrCol_dev[off5+j]], systemMatrix_dev[off1+j]*yi);
            }
        }
}


__global__ void cblas_dcopy_kernel (const long nunkSplit, double* __restrict__ vVect_dev, double* __restrict__ wVect_dev)
{
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(ix < nunkSplit)  wVect_dev[ix] = vVect_dev[ix];
}




// ---------------------------------------------------------------------
// d2norm  returns  sqrt( a**2 + b**2 )  with precautions
// to avoid overflow.
//
// 21 Mar 1990: First version.
// ---------------------------------------------------------------------
static inline double
d2norm( const double a, const double b )
{
    double scale;
    const double zero = 0.0;

    scale  = fabs( a ) + fabs( b );
    if (scale == zero)
        return zero;
    else
        return scale * sqrt( (a/scale)*(a/scale) + (b/scale)*(b/scale) );
}

static inline void
dload(const long n, const double alpha, double x[] )
{    
    #pragma omp for
    for (long i = 0; i < n; i++) x[i] = alpha;
    return;
}


inline long create_startend_gpulist(const long* matrixIndexAstro,long* startend_dev, const long mapNoss, const long nnz){
    long *startend=(long*)malloc(sizeof(long)*(nnz+1));
    long count=0, nnz2=0;
    startend[nnz2]=count;nnz2++;
    for(long i=0; i<mapNoss-1; ++i){
        if(matrixIndexAstro[i]!=matrixIndexAstro[i+1]){
            count++;
            startend[nnz2]=count;
            nnz2++;
        }else{
            count++;
        }
    }
    startend[nnz2]=count+1;

    checkCuda( cudaMemcpy(startend_dev, startend, sizeof(long)*(nnz2+1), cudaMemcpyHostToDevice));

    free(startend);
    return nnz2;
}

// ---------------------------------------------------------------------
// LSQR
// ---------------------------------------------------------------------
void lsqr(
          long int m,
          long int n,
          double damp,
          double *knownTerms,     // len = m  reported as u
          double *vVect,     // len = n reported as v
          double *wVect,     // len = n  reported as w
          double *xSolution,     // len = n reported as x
          double *standardError,    // len at least n.  May be NULL. reported as se
          double atol,
          double btol,
          double conlim,
          int    itnlim,
          // The remaining variables are output only.
          int    *istop_out,
          int    *itn_out,
          double *anorm_out,
          double *acond_out,
          double *rnorm_out,
          double *arnorm_out,
          double *xnorm_out,
          double *sysmatAstro,
          double *sysmatAtt,
          double *sysmatInstr,
          double *sysmatGloB,
          double *sysmatConstr,
          long *matrixIndexAstro, // reported as janew
          long *matrixIndexAtt, // reported as janew
          int *instrCol,
          int *instrConstrIlung,
        //   double *preCondVect, 
	  struct comData comlsqr){ 
//     ------------------------------------------------------------------
//
//     LSQR  finds a solution x to the following problems:
//
//     1. Unsymmetric equations --    solve  A*x = b
//
//     2. Linear least squares  --    solve  A*x = b
//                                    in the least-squares sense
//
//     3. Damped least squares  --    solve  (   A    )*x = ( b )
//                                           ( damp*I )     ( 0 )
//                                    in the least-squares sense
//
//     where A is a matrix with m rows and n columns, b is an
//     m-vector, and damp is a scalar.  (All quantities are real.)
//     The matrix A is intended to be large and sparse.  It is accessed
//     by means of subroutine calls of the form
//
//                aprod ( mode, m, n, x, y, UsrWrk )
//
//     which must perform the following functions:
//
//                If mode = 1, compute  y = y + A*x.
//                If mode = 2, compute  x = x + A(transpose)*y.
//
//     The vectors x and y are input parameters in both cases.
//     If  mode = 1,  y should be altered without changing x.
//     If  mode = 2,  x should be altered without changing y.
//     The parameter UsrWrk may be used for workspace as described
//     below.
//
//     The rhs vector b is input via u, and subsequently overwritten.
//
//
//     Note:  LSQR uses an iterative method to approximate the solution.
//     The number of iterations required to reach a certain accuracy
//     depends strongly on the scaling of the problem.  Poor scaling of
//     the rows or columns of A should therefore be avoided where
//     possible.
//
//     For example, in problem 1 the solution is unaltered by
//     row-scaling.  If a row of A is very small or large compared to
//     the other rows of A, the corresponding row of ( A  b ) should be
//     scaled up or down.
//
//     In problems 1 and 2, the solution x is easily recovered
//     following column-scaling.  Unless better information is known,
//     the nonzero columns of A should be scaled so that they all have
//     the same Euclidean norm (e.g., 1.0).
//
//     In problem 3, there is no freedom to re-scale if damp is
//     nonzero.  However, the value of damp should be assigned only
//     after attention has been paid to the scaling of A.
//
//     The parameter damp is intended to help regularize
//     ill-conditioned systems, by preventing the true solution from
//     being very large.  Another aid to regularization is provided by
//     the parameter acond, which may be used to terminate iterations
//     before the computed solution becomes very large.
//
//     Note that x is not an input parameter.
//     If some initial estimate x0 is known and if damp = 0,
//     one could proceed as follows:
//
//       1. Compute a residual vector     r0 = b - A*x0.
//       2. Use LSQR to solve the system  A*dx = r0.
//       3. Add the correction dx to obtain a final solution x = x0 + dx.
//
//     This requires that x0 be available before and after the call
//     to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations
//     to solve A*x = b and k2 iterations to solve A*dx = r0.
//     If x0 is "good", norm(r0) will be smaller than norm(b).
//     If the same stopping tolerances atol and btol are used for each
//     system, k1 and k2 will be similar, but the final solution x0 + dx
//     should be more accurate.  The only way to reduce the total work
//     is to use a larger stopping tolerance for the second system.
//     If some value btol is suitable for A*x = b, the larger value
//     btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.
//
//     Preconditioning is another way to reduce the number of iterations.
//     If it is possible to solve a related system M*x = b efficiently,
//     where M approximates A in some helpful way
//     (e.g. M - A has low rank or its elements are small relative to
//     those of A), LSQR may converge more rapidly on the system
//           A*M(inverse)*z = b,
//     after which x can be recovered by solving M*x = z.
//
//     NOTE: If A is symmetric, LSQR should not be used!
//     Alternatives are the symmetric conjugate-gradient method (cg)
//     and/or SYMMLQ.
//     SYMMLQ is an implementation of symmetric cg that applies to
//     any symmetric A and will converge more rapidly than LSQR.
//     If A is positive definite, there are other implementations of
//     symmetric cg that require slightly less work per iteration
//     than SYMMLQ (but will take the same number of iterations).
//
//
//     Notation
//     --------
//
//     The following quantities are used in discussing the subroutine
//     parameters:
//
//     Abar   =  (   A    ),          bbar  =  ( b )
//               ( damp*I )                    ( 0 )
//
//     r      =  b  -  A*x,           rbar  =  bbar  -  Abar*x
//
//     rnorm  =  sqrt( norm(r)**2  +  damp**2 * norm(x)**2 )
//            =  norm( rbar )
//
//     relpr  =  the relative precision of floating-point arithmetic
//               on the machine being used.  On most machines,
//               relpr is about 1.0e-7 and 1.0d-16 in single and double
//               precision respectively.
//
//     LSQR  minimizes the function rnorm with respect to x.
//
//
//     Parameters
//     ----------
//
//     m       input      m, the number of rows in A.
//
//     n       input      n, the number of columns in A.
//
//     aprod   external   See above.
//
//     damp    input      The damping parameter for problem 3 above.
//                        (damp should be 0.0 for problems 1 and 2.)
//                        If the system A*x = b is incompatible, values
//                        of damp in the range 0 to sqrt(relpr)*norm(A)
//                        will probably have a negligible effect.
//                        Larger values of damp will tend to decrease
//                        the norm of x and reduce the number of 
//                        iterations required by LSQR.
//
//                        The work per iteration and the storage needed
//                        by LSQR are the same for all values of damp.
//
//     rw      workspace  Transit pointer to user's workspace.
//                        Note:  LSQR  does not explicitly use this
//                        parameter, but passes it to subroutine aprod for
//                        possible use as workspace.
//
//     u(m)    input      The rhs vector b.  Beware that u is
//                        over-written by LSQR.
//
//     v(n)    workspace
//
//     w(n)    workspace
//
//     x(n)    output     Returns the computed solution x.
//
//     se(*)   output     If m .gt. n  or  damp .gt. 0,  the system is
//             (maybe)    overdetermined and the standard errors may be
//                        useful.  (See the first LSQR reference.)
//                        Otherwise (m .le. n  and  damp = 0) they do not
//                        mean much.  Some time and storage can be saved
//                        by setting  se = NULL.  In that case, se will
//                        not be touched.
//
//                        If se is not NULL, then the dimension of se must
//                        be n or more.  se(1:n) then returns standard error
//                        estimates for the components of x.
//                        For each i, se(i) is set to the value
//                           rnorm * sqrt( sigma(i,i) / t ),
//                        where sigma(i,i) is an estimate of the i-th
//                        diagonal of the inverse of Abar(transpose)*Abar
//                        and  t = 1      if  m .le. n,
//                             t = m - n  if  m .gt. n  and  damp = 0,
//                             t = m      if  damp .ne. 0.
//
//     atol    input      An estimate of the relative error in the data
//                        defining the matrix A.  For example,
//                        if A is accurate to about 6 digits, set
//                        atol = 1.0e-6 .
//
//     btol    input      An estimate of the relative error in the data
//                        defining the rhs vector b.  For example,
//                        if b is accurate to about 6 digits, set
//                        btol = 1.0e-6 .
//
//     conlim  input      An upper limit on cond(Abar), the apparent
//                        condition number of the matrix Abar.
//                        Iterations will be terminated if a computed
//                        estimate of cond(Abar) exceeds conlim.
//                        This is intended to prevent certain small or
//                        zero singular values of A or Abar from
//                        coming into effect and causing unwanted growth
//                        in the computed solution.
//
//                        conlim and damp may be used separately or
//                        together to regularize ill-conditioned systems.
//
//                        Normally, conlim should be in the range
//                        1000 to 1/relpr.
//                        Suggested value:
//                        conlim = 1/(100*relpr)  for compatible systems,
//                        conlim = 1/(10*sqrt(relpr)) for least squares.
//
//             Note:  If the user is not concerned about the parameters
//             atol, btol and conlim, any or all of them may be set
//             to zero.  The effect will be the same as the values
//             relpr, relpr and 1/relpr respectively.
//
//     itnlim  input      An upper limit on the number of iterations.
//                        Suggested value:
//                        itnlim = n/2   for well-conditioned systems
//                                       with clustered singular values,
//                        itnlim = 4*n   otherwise.
//
//     nout    input      File number for printed output.  If positive,
//                        a summary will be printed on file nout.
//
//     istop   output     An integer giving the reason for termination:
//
//                0       x = 0  is the exact solution.
//                        No iterations were performed.
//
//                1       The equations A*x = b are probably
//                        compatible.  Norm(A*x - b) is sufficiently
//                        small, given the values of atol and btol.
//
//                2       damp is zero.  The system A*x = b is probably
//                        not compatible.  A least-squares solution has
//                        been obtained that is sufficiently accurate,
//                        given the value of atol.
//
//                3       damp is nonzero.  A damped least-squares
//                        solution has been obtained that is sufficiently
//                        accurate, given the value of atol.
//
//                4       An estimate of cond(Abar) has exceeded
//                        conlim.  The system A*x = b appears to be
//                        ill-conditioned.  Otherwise, there could be an
//                        error in subroutine aprod.
//
//                5       The iteration limit itnlim was reached.
//
//     itn     output     The number of iterations performed.
//
//     anorm   output     An estimate of the Frobenius norm of  Abar.
//                        This is the square-root of the sum of squares
//                        of the elements of Abar.
//                        If damp is small and if the columns of A
//                        have all been scaled to have length 1.0,
//                        anorm should increase to roughly sqrt(n).
//                        A radically different value for anorm may
//                        indicate an error in subroutine aprod (there
//                        may be an inconsistency between modes 1 and 2).
//
//     acond   output     An estimate of cond(Abar), the condition
//                        number of Abar.  A very high value of acond
//                        may again indicate an error in aprod.
//
//     rnorm   output     An estimate of the final value of norm(rbar),
//                        the function being minimized (see notation
//                        above).  This will be small if A*x = b has
//                        a solution.
//
//     arnorm  output     An estimate of the final value of
//                        norm( Abar(transpose)*rbar ), the norm of
//                        the residual for the usual normal equations.
//                        This should be small in all cases.  (arnorm
//                        will often be smaller than the true value
//                        computed from the output vector x.)
//
//     xnorm   output     An estimate of the norm of the final
//                        solution vector x.
//
//
//     Subroutines and functions used              
//     ------------------------------
//
//     USER               aprod
//     CBLAS              dcopy, dnrm2, dscal (see Lawson et al. below)
//
//
//     References
//     ----------
//
//     C.C. Paige and M.A. Saunders,  LSQR: An algorithm for sparse
//          linear equations and sparse least squares,
//          ACM Transactions on Mathematical Software 8, 1 (March 1982),
//          pp. 43-71.
//
//     C.C. Paige and M.A. Saunders,  Algorithm 583, LSQR: Sparse
//          linear equations and least-squares problems,
//          ACM Transactions on Mathematical Software 8, 2 (June 1982),
//          pp. 195-209.
//
//     C.L. Lawson, R.J. Hanson, D.R. Kincaid and F.T. Krogh,
//          Basic linear algebra subprograms for Fortran usage,
//          ACM Transactions on Mathematical Software 5, 3 (Sept 1979),
//          pp. 308-323 and 324-325.
//     ------------------------------------------------------------------
//
//
//     LSQR development:
//     22 Feb 1982: LSQR sent to ACM TOMS to become Algorithm 583.
//     15 Sep 1985: Final F66 version.  LSQR sent to "misc" in netlib.
//     13 Oct 1987: Bug (Robert Davies, DSIR).  Have to delete
//                     if ( (one + dabs(t)) .le. one ) GO TO 200
//                  from loop 200.  The test was an attempt to reduce
//                  underflows, but caused w(i) not to be updated.
//     17 Mar 1989: First F77 version.
//     04 May 1989: Bug (David Gay, AT&T).  When the second beta is zero,
//                  rnorm = 0 and
//                  test2 = arnorm / (anorm * rnorm) overflows.
//                  Fixed by testing for rnorm = 0.
//     05 May 1989: Sent to "misc" in netlib.
//     14 Mar 1990: Bug (John Tomlin via IBM OSL testing).
//                  Setting rhbar2 = rhobar**2 + dampsq can give zero
//                  if rhobar underflows and damp = 0.
//                  Fixed by testing for damp = 0 specially.
//     15 Mar 1990: Converted to lower case.
//     21 Mar 1990: d2norm introduced to avoid overflow in numerous
//                  items like  c = sqrt( a**2 + b**2 ).
//     04 Sep 1991: wantse added as an argument to LSQR, to make
//                  standard errors optional.  This saves storage and
//                  time when se(*) is not wanted.
//     13 Feb 1992: istop now returns a value in [1,5], not [1,7].
//                  1, 2 or 3 means that x solves one of the problems
//                  Ax = b,  min norm(Ax - b)  or  damped least squares.
//                  4 means the limit on cond(A) was reached.
//                  5 means the limit on iterations was reached.
//     07 Dec 1994: Keep track of dxmax = max_k norm( phi_k * d_k ).
//                  So far, this is just printed at the end.
//                  A large value (relative to norm(x)) indicates
//                  significant cancellation in forming
//                  x  =  D*f  =  sum( phi_k * d_k ).
//                  A large column of D need NOT be serious if the
//                  corresponding phi_k is small.
//     27 Dec 1994: Include estimate of alfa_opt in iteration log.
//                  alfa_opt is the optimal scale factor for the
//                  residual in the "augmented system", as described by
//                  A. Bjorck (1992),
//                  Pivoting and stability in the augmented system method,
//                  in D. F. Griffiths and G. A. Watson (eds.),
//                  "Numerical Analysis 1991",
//                  Proceedings of the 14th Dundee Conference,
//                  Pitman Research Notes in Mathematics 260,
//                  Longman Scientific and Technical, Harlow, Essex, 1992.
//     14 Apr 2006: "Line-by-line" conversion to ISO C by
//                  Michael P. Friedlander.
//
//
//     Michael A. Saunders                  mike@sol-michael.stanford.edu
//     Dept of Operations Research          na.Msaunders@na-net.ornl.gov
//     Stanford University
//     Stanford, CA 94305-4022              (415) 723-1875
//-----------------------------------------------------------------------

//  Local copies of output variables.  Output vars are assigned at exit.
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: TIME 1
    double communicationtime=ZERO, timeiter=ZERO, starttime=ZERO, GlobalTime=ZERO, TimeSetInit=ZERO, LoopTime=ZERO;
    MPI_Barrier(MPI_COMM_WORLD);
    TimeSetInit=MPI_Wtime();
    GlobalTime=MPI_Wtime();

    
    const int myid=comlsqr.myid;
    const long mapNoss=static_cast<long>(comlsqr.mapNoss[myid]);

    int deviceCount = 0;
    checkCuda( cudaGetDeviceCount(&deviceCount) );
    checkCuda( cudaSetDevice(myid % deviceCount) );
    int deviceNum = 0;
    checkCuda( cudaGetDevice(&deviceNum) );

    // printf("PE = %d, deviceNum = %d\n", myid, deviceNum);

    int
        istop  = 0,
        itn    = 0;
    double
        anorm  = ZERO,
        acond  = ZERO,
        rnorm  = ZERO,
        arnorm = ZERO,
        xnorm  = ZERO;
       
    //  Local variables
    const bool
        damped = damp > ZERO,
        wantse = standardError != NULL;

    double
        alpha, beta, bnorm,
        cs, cs1, cs2, ctol,
        delta, dknorm, dnorm, dxk, dxmax,
        gamma, gambar, phi, phibar, psi,
        res2, rho, rhobar, rhbar1,
        rhs, rtol, sn, sn1, sn2,
        t, tau, temp, test1, test3,
        theta, t1, t2, t3, xnorm1, z, zbar;
    double test2=0;
    
    //-----------------------------------------------------------------------


    ///////////// Specific definitions
    double startCycleTime,endCycleTime,totTime; // ompSec=0.0;
    long  other; 
    int nAstroElements;
    
    ////////////////////////////////	
    //  Initialize.
    const long VrIdAstroPDimMax=comlsqr.VrIdAstroPDimMax; 
    const long VrIdAstroPDim=comlsqr.VrIdAstroPDim;  
    const long nDegFreedomAtt=comlsqr.nDegFreedomAtt;
    const long localAstro=VrIdAstroPDim*comlsqr.nAstroPSolved;
    const long offsetAttParam = comlsqr.offsetAttParam;
    const long offsetInstrParam = comlsqr.offsetInstrParam;
    const long offsetGlobParam = comlsqr.offsetGlobParam;  
    const long nunkSplit=comlsqr.nunkSplit;
    const long offLocalAstro = comlsqr.mapStar[myid][0] * comlsqr.nAstroPSolved;
    const long localAstroMax = VrIdAstroPDimMax * comlsqr.nAstroPSolved; 
    const long offLocalInstr = offsetInstrParam + (localAstroMax - offsetAttParam); 
    const long offLocalGlob = offsetGlobParam + (localAstroMax - offsetAttParam); 
    const long offLocalAtt = localAstroMax - offsetAttParam; 
    
    const int nEqExtConstr=comlsqr.nEqExtConstr;
    const int nEqBarConstr=comlsqr.nEqBarConstr;
    const int nOfInstrConstr=comlsqr.nOfInstrConstr;

    const int nproc=comlsqr.nproc;
    const int nAttParam=comlsqr.nAttParam; 
    const int nInstrParam=comlsqr.nInstrParam; 
    const int nGlobalParam=comlsqr.nGlobalParam; 
    const int numOfExtStar=comlsqr.numOfExtStar;
    const int numOfBarStar=comlsqr.numOfBarStar;
    const int numOfExtAttCol=comlsqr.numOfExtAttCol;
    const int nElemIC=comlsqr.nElemIC;
    const int nOfElextObs = comlsqr.nOfElextObs;
    const int nOfElBarObs = comlsqr.nOfElBarObs;
    const int startingAttColExtConstr = comlsqr.startingAttColExtConstr;


    const short nAttP=comlsqr.nAttP;
    const short nAttAxes=comlsqr.nAttAxes;
    const short nInstrPSolved=comlsqr.nInstrPSolved;
    const short nAttParAxis = comlsqr.nAttParAxis;
    const short nAstroPSolved=comlsqr.nAstroPSolved;
    const short nGlobP = comlsqr.nGlobP;


    double alphaLoc2;

    long nElemKnownTerms = mapNoss+nEqExtConstr+nEqBarConstr+nOfInstrConstr;
    long nTotConstraints  =   nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr+nElemIC;
    ///////////// CUDA definitions

    double max_knownTerms;
    double ssq_knownTerms;
    double max_vVect;  
    double ssq_vVect;

    double  *sysmatAstro_dev        =nullptr;
    double  *sysmatAtt_dev          =nullptr;
    double  *sysmatInstr_dev        =nullptr;
    double  *sysmatGloB_dev         =nullptr;
    double  *sysmatConstr_dev       =nullptr;
    double  *vVect_dev              =nullptr;
    double  *knownTerms_dev         =nullptr; 
    double  *wVect_dev              =nullptr;
    double  *kAuxcopy_dev           =nullptr;
    double  *vAuxVect_dev           =nullptr;
    double  *xSolution_dev          =nullptr;
    double  *standardError_dev      =nullptr;


    double *dev_vVect_glob_sum      =nullptr;
    double *dev_max_knownTerms      =nullptr;
    double *dev_ssq_knownTerms      =nullptr;
    double *dev_max_vVect           =nullptr;
    double *dev_ssq_vVect           =nullptr;

            
    long    *matrixIndexAstro_dev   =nullptr;
    long    *matrixIndexAtt_dev     =nullptr;
    long    *startend_dev           =nullptr;

    int     *instrCol_dev           =nullptr; 
    int     *instrConstrIlung_dev   =nullptr;

    //--------------------------------------------------------------------------------------------------------

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaStream_t streamAprod2_0;
    cudaStream_t streamAprod2_1;
    cudaStream_t streamAprod2_2;
    cudaStream_t streamAprod2_3;
    cudaStream_t streamAprod2_4;
    cudaStream_t streamAprod2_5;
    cudaStream_t streamAprod2_6;
    cudaStream_t streamAprod2_7;
    cudaStreamCreate(&streamAprod2_0);
    cudaStreamCreate(&streamAprod2_1);
    cudaStreamCreate(&streamAprod2_2);
    cudaStreamCreate(&streamAprod2_3);
    cudaStreamCreate(&streamAprod2_4);
    cudaStreamCreate(&streamAprod2_5);
    cudaStreamCreate(&streamAprod2_6);
    cudaStreamCreate(&streamAprod2_7);



    checkCuda(cudaMalloc((void**)&sysmatAstro_dev, mapNoss*nAstroPSolved*sizeof(double)) );
    checkCuda(cudaMalloc((void**)&sysmatAtt_dev, mapNoss*nAttP*sizeof(double)) );
    checkCuda(cudaMalloc((void**)&sysmatInstr_dev, mapNoss*nInstrPSolved*sizeof(double)) );
    checkCuda(cudaMalloc((void**)&sysmatGloB_dev, mapNoss*nGlobP*sizeof(double)) );
    checkCuda(cudaMalloc((void**)&sysmatConstr_dev, (nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr+nElemIC)*sizeof(double)) );
    
    checkCuda(cudaMemcpyAsync(sysmatAstro_dev, sysmatAstro, mapNoss*nAstroPSolved*sizeof(double), cudaMemcpyHostToDevice,stream) );
    checkCuda(cudaMemcpyAsync(sysmatAtt_dev, sysmatAtt, mapNoss*nAttP*sizeof(double), cudaMemcpyHostToDevice,stream) );
    checkCuda(cudaMemcpyAsync(sysmatInstr_dev, sysmatInstr, mapNoss*nInstrPSolved*sizeof(double), cudaMemcpyHostToDevice,stream) );
    checkCuda(cudaMemcpyAsync(sysmatGloB_dev, sysmatGloB, mapNoss*nGlobP*sizeof(double), cudaMemcpyHostToDevice,stream) );
    checkCuda(cudaMemcpyAsync(sysmatConstr_dev, sysmatConstr, (nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr+nElemIC)*sizeof(double), cudaMemcpyHostToDevice,stream) );


    // New list    
    long nnz=1;
    for(long i=0; i<mapNoss-1; i++){
        if(matrixIndexAstro[i]!=matrixIndexAstro[i+1]){
            nnz++;
        }
    }


    checkCuda(cudaMalloc((void**)&startend_dev, sizeof(long)*(nnz+1)));  
    nnz=create_startend_gpulist(matrixIndexAstro,startend_dev,mapNoss,nnz);


    checkCuda( cudaMalloc((void**)&matrixIndexAstro_dev, mapNoss*sizeof(long)));
    checkCuda( cudaMalloc((void**)&matrixIndexAtt_dev, mapNoss*sizeof(long)));      
    checkCuda( cudaMemcpyAsync(matrixIndexAstro_dev, matrixIndexAstro, mapNoss*sizeof(long), cudaMemcpyHostToDevice,stream));
    checkCuda( cudaMemcpyAsync(matrixIndexAtt_dev, matrixIndexAtt, mapNoss*sizeof(long), cudaMemcpyHostToDevice,stream));
    //--------------------------------------------------------------------------------------------------------

    checkCuda(cudaMalloc((void**)&vVect_dev, nunkSplit*sizeof(double)) );
    checkCuda(cudaMalloc((void**)&knownTerms_dev, nElemKnownTerms*sizeof(double)) );
    checkCuda(cudaMalloc((void**)&wVect_dev, nunkSplit*sizeof(double)) );
    checkCuda(cudaMalloc((void**)&kAuxcopy_dev, (nEqExtConstr+nEqBarConstr+nOfInstrConstr)*sizeof(double)) );
    checkCuda(cudaMalloc((void**)&vAuxVect_dev, localAstroMax*sizeof(double)) );
    checkCuda(cudaMalloc((void**)&instrCol_dev,(nInstrPSolved*mapNoss+nElemIC)*sizeof(int)) );  // nobs -> mapNoss.
    checkCuda(cudaMalloc((void**)&instrConstrIlung_dev,nOfInstrConstr*sizeof(int)) );
    checkCuda(cudaMalloc((void**)&xSolution_dev, nunkSplit*sizeof(double)) );
    checkCuda(cudaMalloc((void**)&standardError_dev, nunkSplit*sizeof(double)) );


    for(auto i=0; i<nunkSplit;++i){
        xSolution[i]=0.0;
        standardError[i]=0.0;
    }

    checkCuda (cudaMemcpyAsync(xSolution_dev, xSolution, nunkSplit*sizeof(double), cudaMemcpyHostToDevice,stream) );
    checkCuda (cudaMemcpyAsync(standardError_dev, standardError, nunkSplit*sizeof(double), cudaMemcpyHostToDevice,stream) );


    //  Copies H2D:
    checkCuda(cudaMemcpyAsync(instrCol_dev, instrCol, (nInstrPSolved*mapNoss+nElemIC)*sizeof(int), cudaMemcpyHostToDevice,stream) );  // nobs -> mapNoss.
    checkCuda(cudaMemcpyAsync(instrConstrIlung_dev, instrConstrIlung, nOfInstrConstr*sizeof(int), cudaMemcpyHostToDevice,stream));
    
    /* First copy of knownTerms from the host to the device: */
    checkCuda(cudaMemcpyAsync(knownTerms_dev, knownTerms, nElemKnownTerms*sizeof(double), cudaMemcpyHostToDevice,stream));

    checkCuda(cudaMalloc((void**)&dev_vVect_glob_sum, sizeof(double)*gridSize));
    checkCuda(cudaMalloc((void**)&dev_max_knownTerms, sizeof(double)*gridSize));
    checkCuda(cudaMalloc((void**)&dev_ssq_knownTerms, sizeof(double)*gridSize));
    checkCuda(cudaMalloc((void**)&dev_max_vVect, sizeof(double)*gridSize));
    checkCuda(cudaMalloc((void**)&dev_ssq_vVect, sizeof(double)*gridSize));
    

    double* dknorm_vec;
    cudaMalloc((void**)&dknorm_vec,sizeof(double));

    //  Grid topologies:
    dim3 blockDim(TILE_WIDTH,1,1);
    dim3 gridDim_aprod1((mapNoss - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_aprod1_Plus_Constr((mapNoss + nEqExtConstr + nEqBarConstr + nOfInstrConstr - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_vAuxVect_Kernel((localAstroMax - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_vVect_Put_To_Zero_Kernel((nunkSplit - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_nunk((nunkSplit - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_aprod2((mapNoss - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_kAuxcopy_Kernel(((nEqExtConstr+nEqBarConstr+nOfInstrConstr) - 1)/TILE_WIDTH + 1,1,1);
    
    //  Grid topologies for the constraints sections:
    const int numOfExtStarTimesnAstroPSolved = numOfExtStar*nAstroPSolved;
    int max_numOfExtStarTimesnAstroPSolved_numOfExtAttCol;
    max_numOfExtStarTimesnAstroPSolved_numOfExtAttCol = numOfExtStarTimesnAstroPSolved;
    if (numOfExtStarTimesnAstroPSolved < numOfExtAttCol)
        max_numOfExtStarTimesnAstroPSolved_numOfExtAttCol = numOfExtAttCol;
    
    dim3 gridDim_aprod1_ExtConstr((max_numOfExtStarTimesnAstroPSolved_numOfExtAttCol - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_aprod1_BarConstr((numOfBarStar*nAstroPSolved - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_aprod1_InstrConstr((nOfInstrConstr - 1)/TILE_WIDTH + 1,1,1);
    
    int max_numOfExtStar_numOfExtAttCol;
    max_numOfExtStar_numOfExtAttCol = numOfExtStar;
    if (numOfExtStar < numOfExtAttCol)
        max_numOfExtStar_numOfExtAttCol = numOfExtAttCol;
    
    dim3 gridDim_aprod2_ExtConstr((max_numOfExtStar_numOfExtAttCol - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_aprod2_BarConstr((numOfBarStar - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_aprod2_InstrConstr((nOfInstrConstr - 1)/TILE_WIDTH + 1,1,1);
    

////////////////////////////////////////////// CUDA Definitions END
other=(long)nAttParam + nInstrParam + nGlobalParam; 
comlsqr.itn=itn;
totTime=comlsqr.totSec;


std::cout<<"nAstroPSolved "<<nAstroPSolved<<std::endl;
std::cout<<"nAtt "<<nAttP<<std::endl;
std::cout<<"nInstrPSolved "<<nInstrPSolved<<std::endl;
std::cout<<"NGlobP "<<nGlobP<<std::endl;
std::cout<<"nEqExtConstr "<<nEqExtConstr<<std::endl;
std::cout<<"nEqBarConstr "<<nEqBarConstr<<std::endl;
std::cout<<"nOfInstrConstr "<<nOfInstrConstr<<std::endl;

// MPI_Barrier(MPI_COMM_WORLD);MPI_Abort(MPI_COMM_WORLD,10);


itn    =   0;
istop  =   0;
ctol   =   ZERO;
if (conlim > ZERO) ctol = ONE / conlim;
anorm  =   ZERO;
acond  =   ZERO;
dnorm  =   ZERO;
dxmax  =   ZERO;
res2   =   ZERO;
psi    =   ZERO;
xnorm  =   ZERO;
xnorm1 =   ZERO;
cs2    = - ONE;
sn2    =   ZERO;
z      =   ZERO;

 	
//  ------------------------------------------------------------------
//  Set up the first vectors u and v for the bidiagonalization.
//  These satisfy  beta*u = b,  alpha*v = A(transpose)*u.
//  ------------------------------------------------------------------

dload( nunkSplit, 0.0, vVect );
    
checkCuda (cudaMemcpyAsync(vVect_dev, vVect, nunkSplit*sizeof(double), cudaMemcpyHostToDevice,stream) );
    
dload( nunkSplit, 0.0, xSolution );

if ( wantse )   dload( nunkSplit, 0.0, standardError );

alpha  =   ZERO;
    
// Find the maximum value of the u array
maxCommMultiBlock_double<<<gridSize, blockSize>>>(knownTerms_dev, dev_max_knownTerms, nElemKnownTerms);
maxCommMultiBlock_double<<<1, blockSize>>>(dev_max_knownTerms, dev_max_knownTerms, gridSize);
checkCuda (cudaMemcpyAsync(&max_knownTerms, dev_max_knownTerms, sizeof(double), cudaMemcpyDeviceToHost,stream) );

double betaLoc, betaLoc2;
if (myid == 0) {
    sumCommMultiBlock_double<<<gridSize, blockSize>>>(knownTerms_dev,dev_ssq_knownTerms,max_knownTerms,nElemKnownTerms);
    realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_knownTerms,dev_ssq_knownTerms,gridSize);
    checkCuda(cudaMemcpyAsync(&ssq_knownTerms, dev_ssq_knownTerms, sizeof(double), cudaMemcpyDeviceToHost,stream) );
    betaLoc = max_knownTerms*sqrt(ssq_knownTerms);
} else {
    sumCommMultiBlock_double<<<gridSize, blockSize>>>(knownTerms_dev,dev_ssq_knownTerms,max_knownTerms,mapNoss);
    realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_knownTerms,dev_ssq_knownTerms,gridSize);
    checkCuda(cudaMemcpyAsync(&ssq_knownTerms, dev_ssq_knownTerms, sizeof(double), cudaMemcpyDeviceToHost,stream) );
    betaLoc = max_knownTerms*sqrt(ssq_knownTerms);
}
    
betaLoc2=betaLoc*betaLoc;
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//------------------------------------------------------------------------------------------------  TIME 2
starttime=MPI_Wtime();
MPI_Allreduce(MPI_IN_PLACE,&betaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
communicationtime+=MPI_Wtime()-starttime;
//------------------------------------------------------------------------------------------------
beta=sqrt(betaLoc2);
    
    
if (beta > ZERO) 
{
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: TIME3

    dscal<<<gridSize,blockSize>>>(knownTerms_dev, 1.0/beta,nElemKnownTerms, 1.0);
	if(myid!=0)
	{
        vVect_Put_To_Zero_Kernel<<<gridDim_vVect_Put_To_Zero_Kernel,TILE_WIDTH>>>(vVect_dev,localAstroMax,nunkSplit);
	}
        
    //APROD2 CALL BEFORE LSQR
    cudaDeviceSynchronize();


    if(nAstroPSolved) aprod2_Kernel_astro<<<BlockXGridAprod2Astro,ThreadsXBlockAprod2Astro,0,streamAprod2_0>>>(vVect_dev, sysmatAstro_dev, knownTerms_dev, matrixIndexAstro_dev, startend_dev, offLocalAstro, nnz, nAstroPSolved);
    if(nAttP) aprod2_Kernel_att_AttAxis<<<BlockXGrid,ThreadsXBlock,0,streamAprod2_1>>>(vVect_dev, sysmatAtt_dev, knownTerms_dev, matrixIndexAtt_dev, nAttP, nDegFreedomAtt, offLocalAtt, mapNoss, nAstroPSolved, nAttParAxis);
    if(nInstrPSolved) aprod2_Kernel_instr<<<BlockXGrid,ThreadsXBlock,0,streamAprod2_2>>>(vVect_dev, sysmatInstr_dev, knownTerms_dev, instrCol_dev, offLocalInstr, mapNoss, nInstrPSolved);
    for (short inGlob = 0; inGlob < nGlobP; inGlob++)
    {
        sumCommMultiBlock_double_aprod2_Kernel_glob<<<gridSize, blockSize,0,streamAprod2_3>>>(dev_vVect_glob_sum, sysmatGloB_dev, knownTerms_dev, vVect_dev, nGlobP, mapNoss, offLocalGlob, inGlob);
        realsumCommMultiBlock_double_aprod2_Kernel_glob<<<1, blockSize,0,streamAprod2_4>>>(vVect_dev,dev_vVect_glob_sum, gridSize, offLocalGlob, inGlob);
    }
    cudaDeviceSynchronize();
    //  CONSTRAINTS OF APROD MODE 2:
    if(nEqExtConstr) aprod2_Kernel_ExtConstr<<<gridDim_aprod2_ExtConstr, TILE_WIDTH,0,streamAprod2_5>>>(vVect_dev,sysmatInstr_dev,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAttAxes,nAstroPSolved);
    if(nEqBarConstr) aprod2_Kernel_BarConstr<<<gridDim_aprod2_BarConstr, TILE_WIDTH,0,streamAprod2_6>>>(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
    if(nOfInstrConstr) aprod2_Kernel_InstrConstr<<<gridDim_aprod2_InstrConstr, TILE_WIDTH,0,streamAprod2_7>>>(vVect_dev,sysmatConstr_dev,knownTerms_dev,instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,nDegFreedomAtt,mapNoss,nEqExtConstr,nEqBarConstr,nOfElextObs,nOfElBarObs,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);

    checkCuda ( cudaMemcpyAsync(vVect, vVect_dev, nunkSplit*sizeof(double), cudaMemcpyDeviceToHost,streamAprod2_7) );

    // cudaDeviceSynchronize();


    
    /* ~~~~~~~~~~~~~~ */
    // checkCuda ( cudaMemcpyAsync(vVect, vVect_dev, nunkSplit*sizeof(double), cudaMemcpyDeviceToHost,stream) );
    cudaDeviceSynchronize();
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    //------------------------------------------------------------------------------------------------  TIME4
    starttime=MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE,&vVect[localAstroMax], nAttParam+nInstrParam+nGlobalParam,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    communicationtime+=MPI_Wtime()-starttime;
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    //------------------------------------------------------------------------------------------------
    if(nAstroPSolved) SumCirc2(vVect,comlsqr,&communicationtime);
    //------------------------------------------------------------------------------------------------
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    
    checkCuda (cudaMemcpyAsync(vVect_dev, vVect, nunkSplit*sizeof(double), cudaMemcpyHostToDevice,stream) );

    nAstroElements=comlsqr.mapStar[myid][1]-comlsqr.mapStar[myid][0] + 1;
 	if(myid<nproc-1)
 	{
 		nAstroElements=comlsqr.mapStar[myid][1]-comlsqr.mapStar[myid][0] +1;
 		if(comlsqr.mapStar[myid][1]==comlsqr.mapStar[myid+1][0]) nAstroElements--;
 	}

    // reset internal state
    maxCommMultiBlock_double<<<gridSize, blockSize>>>(vVect_dev, dev_max_vVect, nunkSplit);
    maxCommMultiBlock_double<<<1, blockSize>>>(dev_max_vVect, dev_max_vVect, gridSize);
        
    cudaMemcpyAsync(&max_vVect, dev_max_vVect, sizeof(double), cudaMemcpyDeviceToHost,stream);
 



    double alphaLoc=0.0;
    
    sumCommMultiBlock_double<<<gridSize, blockSize>>>(vVect_dev, dev_ssq_vVect, max_vVect, nAstroElements*nAstroPSolved);
    realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_vVect, dev_ssq_vVect, gridSize);
    cudaMemcpyAsync(&ssq_vVect, dev_ssq_vVect, sizeof(double), cudaMemcpyDeviceToHost,stream);
    alphaLoc = max_vVect*sqrt(ssq_vVect);
     
    alphaLoc2=alphaLoc*alphaLoc;
	if(myid==0) {
        double alphaOther2 = alphaLoc2;
        sumCommMultiBlock_double<<<gridSize, blockSize>>>(&vVect_dev[localAstroMax], dev_ssq_vVect, max_vVect, nunkSplit - localAstroMax);
        realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_vVect, dev_ssq_vVect, gridSize);
        checkCuda (cudaMemcpyAsync(&ssq_vVect, dev_ssq_vVect, sizeof(double), cudaMemcpyDeviceToHost,stream) );
        alphaLoc = max_vVect*sqrt(ssq_vVect);
        alphaLoc2 = alphaLoc*alphaLoc;
        alphaLoc2 = alphaOther2 + alphaLoc2;
	}


    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    //------------------------------------------------------------------------------------------------  TIME6
    starttime=MPI_Wtime();
	MPI_Allreduce(MPI_IN_PLACE,&alphaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    communicationtime+=MPI_Wtime()-starttime;
    //------------------------------------------------------------------------------------------------
    alpha=sqrt(alphaLoc2);
   }


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: TIME7


    if (alpha > ZERO) 
    {
        dscal<<<gridSize,blockSize>>>(vVect_dev, 1/alpha, nunkSplit, 1.0);
        cblas_dcopy_kernel<<<gridDim_nunk,TILE_WIDTH>>>(nunkSplit,vVect_dev,wVect_dev);
        // cudaDeviceSynchronize();
    }
    
    checkCuda (cudaMemcpyAsync(vVect, vVect_dev, nunkSplit*sizeof(double), cudaMemcpyDeviceToHost,stream) );
    checkCuda (cudaMemcpyAsync(wVect, wVect_dev, nunkSplit*sizeof(double), cudaMemcpyDeviceToHost,stream) );
    checkCuda (cudaMemcpyAsync(knownTerms, knownTerms_dev, nElemKnownTerms*sizeof(double), cudaMemcpyDeviceToHost,stream) );


    arnorm  = alpha * beta;

    if (arnorm == ZERO){
        if (damped  &&  istop == 2) istop = 3;

        checkCuda(cudaFree(vVect_dev));
        checkCuda(cudaFree(wVect_dev));
        checkCuda(cudaFree(knownTerms_dev));
        checkCuda(cudaFree(kAuxcopy_dev));
        checkCuda(cudaFree(vAuxVect_dev));
        checkCuda(cudaFree(instrCol_dev));
        checkCuda(cudaFree(instrConstrIlung_dev));
        checkCuda(cudaFree(dev_vVect_glob_sum));
        checkCuda(cudaFree(dev_max_knownTerms)); 
        checkCuda(cudaFree(dev_ssq_knownTerms)); 
        checkCuda(cudaFree(dev_max_vVect)); 
        checkCuda(cudaFree(dev_ssq_vVect)); 
        checkCuda(cudaFree(matrixIndexAstro_dev));
        checkCuda(cudaFree(startend_dev));

        checkCuda(cudaFree(sysmatAstro_dev));
        checkCuda(cudaFree(sysmatAtt_dev));
        checkCuda(cudaFree(sysmatInstr_dev));
        checkCuda(cudaFree(sysmatGloB_dev));
        checkCuda(cudaFree(sysmatConstr_dev));

        *istop_out  = istop;
        *itn_out    = itn;
        *anorm_out  = anorm;
        *acond_out  = acond;
        *rnorm_out  = rnorm;
        *arnorm_out = test2;
        *xnorm_out  = xnorm;
        
        return;
    }


    rhobar =   alpha;
    phibar =   beta;
    bnorm  =   beta;
    rnorm  =   beta;


    if(!myid){
        test1  = ONE;
        test2  = alpha / beta;
    }

    
    checkCuda (cudaMemcpyAsync(knownTerms_dev, knownTerms, nElemKnownTerms*sizeof(double), cudaMemcpyHostToDevice,stream) );
    checkCuda (cudaMemcpyAsync(vVect_dev, vVect, nunkSplit*sizeof(double), cudaMemcpyHostToDevice,stream) );
    checkCuda (cudaMemcpyAsync(wVect_dev, wVect, nunkSplit*sizeof(double), cudaMemcpyHostToDevice,stream) );
    
	if(myid==0) printf("PE=%d  end restart setup\n",myid);

    //  ==================================================================
    //  Main iteration loop.
    //  ==================================================================
    
    if (myid == 0) printf("LSQR: START ITERATIONS\n");
    ////////////////////////  START ITERATIONS

    
    TimeSetInit=MPI_Wtime()-TimeSetInit;
    LoopTime=startCycleTime=MPI_Wtime(); 
    
    while (1) {
        
        endCycleTime=MPI_Wtime()-startCycleTime;
        startCycleTime=MPI_Wtime();
        timeiter+=endCycleTime;
        totTime=totTime+endCycleTime;

        //------------------------------------------------------------------------------------------------
        starttime=MPI_Wtime();
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast( &itnlim, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast( &comlsqr.itnLimit, 1, MPI_INT, 0, MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //------------------------------------------------------------------------------------------------


        if(myid==0) printf("lsqr: Iteration number %d. Iteration seconds %lf. Global Seconds %lf \n",itn,endCycleTime,totTime);
    
        
        itn    = itn + 1;
        comlsqr.itn=itn;
        //      ------------------------------------------------------------------
        //      Perform the next step of the bidiagonalization to obtain the
        //      next  beta, u, alpha, v.  These satisfy the relations
        //                 beta*u  =  A*v  -  alpha*u,
        //                alpha*v  =  A(transpose)*u  -  beta*v.
        //      ------------------------------------------------------------------
        dscal<<<gridSize,blockSize,0,stream>>>(knownTerms_dev, alpha, nElemKnownTerms, -1.0);
        kAuxcopy_Kernel<<<gridDim_kAuxcopy_Kernel, TILE_WIDTH,0,stream>>>(knownTerms_dev,kAuxcopy_dev,mapNoss,nEqExtConstr+nEqBarConstr+nOfInstrConstr);
        //{ // CONTEXT MODE 1//////////////////////////////////// APROD MODE 1
        if(nAstroPSolved) aprod1_Kernel_astro<<<gridDim_aprod1,TILE_WIDTH,0,stream>>>(knownTerms_dev, sysmatAstro_dev, vVect_dev, matrixIndexAstro_dev, mapNoss, offLocalAstro, nAstroPSolved);
        if(nAttP) aprod1_Kernel_att_AttAxis<<<gridDim_aprod1,TILE_WIDTH,0,stream>>>(knownTerms_dev, sysmatAtt_dev, vVect_dev, matrixIndexAtt_dev, nAttP, mapNoss, nDegFreedomAtt, offLocalAtt, nAttParAxis);                
        if(nInstrPSolved) aprod1_Kernel_instr<<<gridDim_aprod1,TILE_WIDTH,0,stream>>>(knownTerms_dev, sysmatInstr_dev, vVect_dev, instrCol_dev, mapNoss, offLocalInstr, nInstrPSolved);
        if(nGlobP) aprod1_Kernel_glob<<<gridDim_aprod1,TILE_WIDTH,0,stream>>>(knownTerms_dev, sysmatGloB_dev , vVect_dev, offLocalGlob, mapNoss, nGlobP);
        // //        CONSTRAINTS APROD MODE 1        
        if(nEqExtConstr) aprod1_Kernel_ExtConstr<<<gridDim_aprod1_ExtConstr,TILE_WIDTH,0,stream>>>(knownTerms_dev,sysmatConstr_dev,vVect_dev,VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,startingAttColExtConstr,nEqExtConstr,nOfElextObs,numOfExtStar, numOfExtAttCol,nAstroPSolved,nAttAxes);
        if(nEqBarConstr) aprod1_Kernel_BarConstr<<<gridDim_aprod1_BarConstr,TILE_WIDTH,0,stream>>>(knownTerms_dev,sysmatConstr_dev,vVect_dev, nOfElextObs,nOfElBarObs,nEqExtConstr,mapNoss,nEqBarConstr,numOfBarStar,nAstroPSolved);
        if(nOfInstrConstr) aprod1_Kernel_InstrConstr<<<gridDim_aprod1_InstrConstr,TILE_WIDTH,0,stream>>>(knownTerms_dev,sysmatConstr_dev,vVect_dev,instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,mapNoss, nDegFreedomAtt,nOfElextObs,nEqExtConstr,nOfElBarObs,nEqBarConstr,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);
        //}// END CONTEXT MODE 1
        checkCuda (cudaMemcpyAsync(&knownTerms[mapNoss], &knownTerms_dev[mapNoss], (nEqExtConstr+nEqBarConstr+nOfInstrConstr)*sizeof(double), cudaMemcpyDeviceToHost,stream) );
        cudaDeviceSynchronize();
        //------------------------------------------------------------------------------------------------
        starttime=MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE,&knownTerms[mapNoss],nEqExtConstr+nEqBarConstr+nOfInstrConstr,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //------------------------------------------------------------------------------------------------
        checkCuda(cudaMemcpyAsync(&knownTerms_dev[mapNoss], &knownTerms[mapNoss], (nEqExtConstr+nEqBarConstr+nOfInstrConstr)*sizeof(double), cudaMemcpyHostToDevice,stream) );
        kauxsum<<<gridDim_kAuxcopy_Kernel,TILE_WIDTH,0,stream>>>(&knownTerms_dev[mapNoss],kAuxcopy_dev,nEqExtConstr+nEqBarConstr+nOfInstrConstr);
        maxCommMultiBlock_double<<<gridSize, blockSize,0,stream>>>(knownTerms_dev, dev_max_knownTerms, nElemKnownTerms);
        maxCommMultiBlock_double<<<1, blockSize,0,stream>>>(dev_max_knownTerms, dev_max_knownTerms, gridSize);
        cudaMemcpyAsync(&max_knownTerms, dev_max_knownTerms, sizeof(double), cudaMemcpyDeviceToHost,stream);
        


        if(myid==0)
        {
            sumCommMultiBlock_double<<<gridSize, blockSize>>>(knownTerms_dev, dev_ssq_knownTerms, max_knownTerms, mapNoss + nEqExtConstr + nEqBarConstr+nOfInstrConstr);
            realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_knownTerms, dev_ssq_knownTerms, gridSize);
            checkCuda (cudaMemcpyAsync(&ssq_knownTerms, dev_ssq_knownTerms, sizeof(double), cudaMemcpyDeviceToHost,stream) );
            betaLoc = max_knownTerms*sqrt(ssq_knownTerms);
            betaLoc2 = betaLoc*betaLoc;
        }else{
            sumCommMultiBlock_double<<<gridSize, blockSize>>>(knownTerms_dev, dev_ssq_knownTerms, max_knownTerms, mapNoss);
            realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_knownTerms, dev_ssq_knownTerms, gridSize);
            checkCuda (cudaMemcpyAsync(&ssq_knownTerms, dev_ssq_knownTerms, sizeof(double), cudaMemcpyDeviceToHost,stream) );
            betaLoc = max_knownTerms*sqrt(ssq_knownTerms);
            betaLoc2 = betaLoc*betaLoc;
        }

        //------------------------------------------------------------------------------------------------
        starttime=MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE,&betaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //------------------------------------------------------------------------------------------------
        //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        beta=sqrt(betaLoc2);
        //  Accumulate  anorm = || Bk || =  sqrt( sum of  alpha**2 + beta**2 + damp**2 ).
        temp   =   d2norm( alpha, beta );
        temp   =   d2norm( temp , damp );
        anorm  =   d2norm( anorm, temp );

        if (beta > ZERO) {
            dscal<<<gridSize,blockSize,0,stream>>>(knownTerms_dev, 1/beta,nElemKnownTerms, 1.0);
            dscal<<<gridSize,blockSize,0,stream>>>(vVect_dev, beta, nunkSplit, -1.0);
            vAuxVect_Kernel<<<gridDim_vAuxVect_Kernel,TILE_WIDTH,0,stream>>>(vVect_dev, vAuxVect_dev, localAstroMax);
            if (myid != 0) {
                vVect_Put_To_Zero_Kernel<<<gridDim_vVect_Put_To_Zero_Kernel,TILE_WIDTH,0,stream>>>(vVect_dev,localAstroMax,nunkSplit);
            }
            //{ // CONTEXT MODE 2 //////////////////////////////////// APROD MODE 2
            //APROD2 CALL BEFORE LSQR
            cudaDeviceSynchronize();

            if(nAstroPSolved) aprod2_Kernel_astro<<<BlockXGridAprod2Astro,ThreadsXBlockAprod2Astro,0,streamAprod2_0>>>(vVect_dev, sysmatAstro_dev, knownTerms_dev, matrixIndexAstro_dev, startend_dev, offLocalAstro, nnz, nAstroPSolved);
            if(nAttP) aprod2_Kernel_att_AttAxis<<<BlockXGrid,ThreadsXBlock,0,streamAprod2_1>>>(vVect_dev, sysmatAtt_dev, knownTerms_dev, matrixIndexAtt_dev, nAttP, nDegFreedomAtt, offLocalAtt, mapNoss, nAstroPSolved, nAttParAxis);
            if(nInstrPSolved) aprod2_Kernel_instr<<<BlockXGrid,ThreadsXBlock,0,streamAprod2_2>>>(vVect_dev, sysmatInstr_dev, knownTerms_dev, instrCol_dev, offLocalInstr, mapNoss, nInstrPSolved);
            for (short inGlob = 0; inGlob < nGlobP; inGlob++)
            {
                sumCommMultiBlock_double_aprod2_Kernel_glob<<<gridSize, blockSize,0,streamAprod2_3>>>(dev_vVect_glob_sum, sysmatGloB_dev, knownTerms_dev, vVect_dev, nGlobP, mapNoss, offLocalGlob, inGlob);
                realsumCommMultiBlock_double_aprod2_Kernel_glob<<<1, blockSize,0,streamAprod2_4>>>(vVect_dev,dev_vVect_glob_sum, gridSize, offLocalGlob, inGlob);
            }
            cudaDeviceSynchronize();
            //  CONSTRAINTS OF APROD MODE 2:
            if(nEqExtConstr) aprod2_Kernel_ExtConstr<<<gridDim_aprod2_ExtConstr, TILE_WIDTH,0,streamAprod2_5>>>(vVect_dev,sysmatInstr_dev,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAttAxes,nAstroPSolved);
            if(nEqBarConstr) aprod2_Kernel_BarConstr<<<gridDim_aprod2_BarConstr, TILE_WIDTH,0,streamAprod2_6>>>(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
            if(nOfInstrConstr) aprod2_Kernel_InstrConstr<<<gridDim_aprod2_InstrConstr, TILE_WIDTH,0,streamAprod2_7>>>(vVect_dev,sysmatConstr_dev,knownTerms_dev,instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,nDegFreedomAtt,mapNoss,nEqExtConstr,nEqBarConstr,nOfElextObs,nOfElBarObs,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);

            checkCuda ( cudaMemcpyAsync(vVect, vVect_dev, nunkSplit*sizeof(double), cudaMemcpyDeviceToHost,streamAprod2_7) );
            cudaDeviceSynchronize();
            //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            //------------------------------------------------------------------------------------------------
            starttime=MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE,&vVect[localAstroMax], nAttParam+nInstrParam+nGlobalParam,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  
            communicationtime+=MPI_Wtime()-starttime;
            //------------------------------------------------------------------------------------------------
            //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::          
            if(nAstroPSolved) SumCirc2(vVect,comlsqr,&communicationtime);
            //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


            checkCuda (cudaMemcpyAsync(vVect_dev, vVect, nunkSplit*sizeof(double), cudaMemcpyHostToDevice,stream) );
            vaux_sum<<<gridDim_vAuxVect_Kernel,TILE_WIDTH>>>(vVect_dev,vAuxVect_dev,localAstroMax);
            maxCommMultiBlock_double<<<gridSize, blockSize>>>(vVect_dev, dev_max_vVect, nunkSplit);
            maxCommMultiBlock_double<<<1, blockSize>>>(dev_max_vVect, dev_max_vVect, gridSize);
            checkCuda (cudaMemcpyAsync(&max_vVect, dev_max_vVect, sizeof(double), cudaMemcpyDeviceToHost,stream) );
                        
            sumCommMultiBlock_double<<<gridSize, blockSize>>>(vVect_dev, dev_ssq_vVect, max_vVect, nAstroElements*nAstroPSolved);
            realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_vVect, dev_ssq_vVect, gridSize);
            checkCuda (cudaMemcpyAsync(&ssq_vVect, dev_ssq_vVect, sizeof(double), cudaMemcpyDeviceToHost,stream) );
            
            double alphaLoc = 0.0;
            alphaLoc = max_vVect*sqrt(ssq_vVect);
            alphaLoc2=alphaLoc*alphaLoc;
    
            if(myid==0) {
                double alphaOther2 = alphaLoc2;
                sumCommMultiBlock_double<<<gridSize, blockSize>>>(&vVect_dev[localAstroMax], dev_ssq_vVect, max_vVect, nunkSplit - localAstroMax);
                realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_vVect, dev_ssq_vVect, gridSize);
                checkCuda (cudaMemcpyAsync(&ssq_vVect, dev_ssq_vVect, sizeof(double), cudaMemcpyDeviceToHost,stream) );
                alphaLoc = max_vVect*sqrt(ssq_vVect);
                alphaLoc2 = alphaLoc*alphaLoc;
                alphaLoc2 = alphaOther2 + alphaLoc2;
            }

            //------------------------------------------------------------------------------------------------
            starttime=MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE,&alphaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            communicationtime+=MPI_Wtime()-starttime;
            //------------------------------------------------------------------------------------------------

            alpha=sqrt(alphaLoc2);
                    
            if (alpha > ZERO) {
                dscal<<<gridSize,blockSize>>>(vVect_dev, 1/alpha, nunkSplit, 1);
            }


        }


        //      ------------------------------------------------------------------
        //      Use a plane rotation to eliminate the damping parameter.
        //      This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        //      ------------------------------------------------------------------
        rhbar1 = rhobar;
        if ( damped ) {
            rhbar1 = d2norm( rhobar, damp );
            cs1    = rhobar / rhbar1;
            sn1    = damp   / rhbar1;
            psi    = sn1 * phibar;
            phibar = cs1 * phibar;
        }

        //      ------------------------------------------------------------------
        //      Use a plane rotation to eliminate the subdiagonal element (beta)
        //      of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        //      ------------------------------------------------------------------
        rho    =   d2norm( rhbar1, beta );
        cs     =   rhbar1 / rho;
        sn     =   beta   / rho;
        theta  =   sn * alpha;
        rhobar = - cs * alpha;
        phi    =   cs * phibar;
        phibar =   sn * phibar;
        tau    =   sn * phi;

        //      ------------------------------------------------------------------
        //      Update  x, w  and (perhaps) the standard error estimates.
        //      ------------------------------------------------------------------
        t1     =   phi   / rho;
        t2     = - theta / rho;
        t3     =   ONE   / rho;
        // dknorm =   ZERO;

        // for (long  i = 0; i < nAstroElements*nAstroPSolved; i++) {
        //         t      =  wVect[i];
        //         t      = (t3*t)*(t3*t);
        //         dknorm += t;
        // }



        dknorm_compute<<<gridSize,blockSize>>>(dknorm_vec,wVect_dev,0,nAstroElements*nAstroPSolved,t3);



        checkCuda (cudaMemcpy(&dknorm, dknorm_vec, sizeof(double), cudaMemcpyDeviceToHost) );
        //------------------------------------------------------------------------------------------------
        starttime=MPI_Wtime();
 		MPI_Allreduce(MPI_IN_PLACE,&dknorm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //------------------------------------------------------------------------------------------------ 		
        checkCuda (cudaMemcpy(dknorm_vec,&dknorm, sizeof(double), cudaMemcpyHostToDevice) );
  	

        transform1<<<gridSize,blockSize>>>(xSolution_dev,wVect_dev,0,localAstro,t1);
        if(wantse)  transform2<<<gridSize,blockSize>>>(standardError_dev,wVect_dev,0,localAstro,t3);
        transform3<<<gridSize,blockSize>>>(wVect_dev,vVect_dev,0,localAstro,t2);

        transform1<<<gridSize,blockSize>>>(xSolution_dev,wVect_dev,localAstroMax,localAstroMax+other,t1);
        if(wantse)  transform2<<<gridSize,blockSize>>>(standardError_dev,wVect_dev,localAstroMax,localAstroMax+other,t3);
        transform3<<<gridSize,blockSize>>>(wVect_dev,vVect_dev,localAstroMax,localAstroMax+other,t2);

        dknorm_compute<<<gridSize,blockSize>>>(dknorm_vec,wVect_dev,localAstroMax,localAstroMax+other,t3);
        checkCuda (cudaMemcpy(&dknorm, dknorm_vec, sizeof(double), cudaMemcpyDeviceToHost) );




        
        ///////////////////////////
        //      ------------------------------------------------------------------
        //      Monitor the norm of d_k, the update to x.
        //      dknorm = norm( d_k )
        //      dnorm  = norm( D_k ),        where   D_k = (d_1, d_2, ..., d_k )
        //      dxk    = norm( phi_k d_k ),  where new x = x_k + phi_k d_k.
        //      ------------------------------------------------------------------
        dknorm = sqrt( dknorm );
        dnorm  = d2norm( dnorm, dknorm );
        dxk    = fabs( phi * dknorm );
        if (dxmax < dxk ) {
            dxmax   =  dxk;
        }

        //      ------------------------------------------------------------------
        //      Use a plane rotation on the right to eliminate the
        //      super-diagonal element (theta) of the upper-bidiagonal matrix.
        //      Then use the result to estimate  norm(x).
        //      ------------------------------------------------------------------
        delta  =   sn2 * rho;
        gambar = - cs2 * rho;
        rhs    =   phi    - delta * z;
        zbar   =   rhs    / gambar;
        xnorm  =   d2norm( xnorm1, zbar  );
        gamma  =   d2norm( gambar, theta );
        cs2    =   gambar / gamma;
        sn2    =   theta  / gamma;
        z      =   rhs    / gamma;
        xnorm1 =   d2norm( xnorm1, z     );

        //      ------------------------------------------------------------------
        //      Test for convergence.
        //      First, estimate the norm and condition of the matrix  Abar,
        //      and the norms of  rbar  and  Abar(transpose)*rbar.
        //      ------------------------------------------------------------------
        acond  =   anorm * dnorm;
        res2   =   d2norm( res2 , psi    );
        rnorm  =   d2norm( res2 , phibar );
        arnorm =   alpha * fabs( tau );

        //      Now use these norms to estimate certain other quantities,
        //      some of which will be small near a solution.


        test1  =   rnorm /  bnorm;
        test2  =   ZERO;
        if (rnorm   > ZERO) test2 = arnorm / (anorm * rnorm);
        test3  =   ONE   /  acond;
        t1     =   test1 / (ONE  +  anorm * xnorm / bnorm);
        rtol   =   btol  +  atol *  anorm * xnorm / bnorm;

        //      The following tests guard against extremely small values of
        //      atol, btol  or  ctol.  (The user may have set any or all of
        //      the parameters  atol, btol, conlim  to zero.)
        //      The effect is equivalent to the normal tests using
        //      atol = relpr,  btol = relpr,  conlim = 1/relpr.

        t3     =   ONE + test3;
        t2     =   ONE + test2;
        t1     =   ONE + t1;

        if (itn >= itnlim) istop = 5;
        if (t3  <= ONE   ) istop = 4;
        if (t2  <= ONE   ) istop = 2;
        if (t1  <= ONE   ) istop = 1;


        if (test3 <= ctol) istop = 4;
        if (test2 <= atol) istop = 2;
        if (test1 <= rtol) istop = 1;   //(Michael Friedlander had this commented out)


    if (istop) break;



    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


    }
    //  ==================================================================
    //  End of iteration loop.
    //  ==================================================================
    //  Finish off the standard error estimates.

    LoopTime=MPI_Wtime()-LoopTime;


    MPI_Barrier(MPI_COMM_WORLD);
    GlobalTime=MPI_Wtime()-GlobalTime;
    //------------------------------------------------------------------------------------------------
    MPI_Allreduce(MPI_IN_PLACE, &communicationtime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(!myid) printf("Max Communication time: %lf \n",communicationtime);
    MPI_Allreduce(MPI_IN_PLACE, &TimeSetInit, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(!myid) printf("Max Before Loop time: %lf \n",TimeSetInit);
    MPI_Allreduce(MPI_IN_PLACE, &LoopTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(!myid) printf("Max Loop time: %lf \n",LoopTime);
    double maxavtime=timeiter/itn;
    if(!myid) printf("Average iteration time: %lf \n", timeiter/itn);
    MPI_Allreduce(MPI_IN_PLACE, &maxavtime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(!myid) printf("Max Average iteration time: %lf \n",maxavtime);
    MPI_Allreduce(MPI_IN_PLACE, &GlobalTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(!myid) printf("Max Total Time time: %lf \n",GlobalTime);


    //------------------------------------------------------------------------------------------------ 		

    checkCuda (cudaMemcpy(xSolution,xSolution_dev, nunkSplit*sizeof(double), cudaMemcpyDeviceToHost) );
    checkCuda (cudaMemcpy(standardError,standardError_dev, nunkSplit*sizeof(double), cudaMemcpyDeviceToHost) );

    if ( wantse ) {
        t    =   ONE;
        if (m > n)     t = m - n;
        if ( damped )  t = m;
        t    =   rnorm / sqrt( t );
      
        for (long i = 0; i < nunkSplit; i++)
            standardError[i]  = t * sqrt( standardError[i] );
        
    }





    //  Assign output variables from local copies.
    *istop_out  = istop;
    *itn_out    = itn;
    *anorm_out  = anorm;
    *acond_out  = acond;
    *rnorm_out  = rnorm;
    *arnorm_out = test2;
    *xnorm_out  = xnorm;


    checkCuda(cudaFree(xSolution_dev));
    checkCuda(cudaFree(standardError_dev));
    checkCuda(cudaFree(dknorm_vec));

    

    checkCuda(cudaFree(vVect_dev));
    checkCuda(cudaFree(wVect_dev));
    checkCuda(cudaFree(knownTerms_dev));
    checkCuda(cudaFree(kAuxcopy_dev));
    checkCuda(cudaFree(vAuxVect_dev));
    checkCuda(cudaFree(instrCol_dev));
    checkCuda(cudaFree(instrConstrIlung_dev));
    checkCuda(cudaFree(dev_vVect_glob_sum));
    checkCuda(cudaFree(dev_max_knownTerms)); 
    checkCuda(cudaFree(dev_ssq_knownTerms)); 
    checkCuda(cudaFree(dev_max_vVect)); 
    checkCuda(cudaFree(dev_ssq_vVect)); 
    checkCuda(cudaFree(matrixIndexAstro_dev));
    checkCuda(cudaFree(startend_dev));

    checkCuda(cudaFree(sysmatAstro_dev));
    checkCuda(cudaFree(sysmatAtt_dev));
    checkCuda(cudaFree(sysmatInstr_dev));
    checkCuda(cudaFree(sysmatGloB_dev));
    checkCuda(cudaFree(sysmatConstr_dev));

    cudaStreamDestroy(stream);
    cudaStreamDestroy(streamAprod2_0);
    cudaStreamDestroy(streamAprod2_1);
    cudaStreamDestroy(streamAprod2_2);
    cudaStreamDestroy(streamAprod2_3);
    cudaStreamDestroy(streamAprod2_4);
    cudaStreamDestroy(streamAprod2_5);
    cudaStreamDestroy(streamAprod2_6);
    cudaStreamDestroy(streamAprod2_7);

    return;
}


