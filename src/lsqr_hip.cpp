/* 

HIP Version

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
#include "hip/hip_runtime.h"



inline
hipError_t checkHip(hipError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != hipSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            hipGetErrorString(result));
    assert(result == hipSuccess);
  }
#endif
  return result;
}



#define ZERO   0.0
#define ONE    1.0

static const int blockSize = 128;
static const int gridSize = 1024;



#if defined(__AMD__)
    #define ThreadsXBlock 32
    #define	BlockXGrid 128	
    #define ThreadsXBlockAprod2Astro 32	
    #define BlockXGridAprod2Astro 64	
    #define TILE_WIDTH 32

    #define atomicAdd(x, y) (__hip_atomic_fetch_add(x, y, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))

#elif defined(__NVIDIA90__)
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

__global__ void  kAuxcopy_Kernel(double* __restrict__  knownTerms_dev, double* __restrict__  kAuxcopy_dev, const long nobs, const int N)
{
    const long ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ix < N) {
        kAuxcopy_dev[ix] = knownTerms_dev[nobs + ix];
        knownTerms_dev[nobs + ix] = 0.0;
    }
}


__global__ void vAuxVect_Kernel(double* __restrict__ vVect_dev, double* __restrict__  vAuxVect_dev, const long N)
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


__global__ void kauxsum (double*  __restrict__ knownTerms_dev,double*  __restrict__ kAuxcopy_dev, const int n)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ix < n) {
        knownTerms_dev[ix] = knownTerms_dev[ix]+kAuxcopy_dev[ix];
    }
}


__global__ void vaux_sum(double*  __restrict__ vV, double*  __restrict__ vA, const long lAM)
{
    const long ix = blockIdx.x * blockDim.x + threadIdx.x;

    if(ix<lAM){
        vV[ix]+=vA[ix];
    }
} 

__global__ void transform1(double*  __restrict__ xSolution, const double* __restrict__  wVect, const long begin, const long end, const double t1){
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

__global__ void transform3(double*  __restrict__ wVect, const double* __restrict__  vVect, const long begin, const long end, const double t2){
    long ix = blockIdx.x * blockDim.x + threadIdx.x+begin;

    while(ix < end){
        wVect[ix]   =  vVect[ix]+t2*wVect[ix];
        ix+=gridDim.x*blockDim.x;
    }
}

__global__ void cblas_dcopy_kernel (const long nunkSplit, double*  __restrict__ vVect_dev, double* __restrict__  wVect_dev)
{
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(ix < nunkSplit)  wVect_dev[ix] = vVect_dev[ix];
}


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


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


// Astrometric part
__global__ void aprod1_Kernel_astro(double*  __restrict__ knownTerms_dev,
                                    const double*  __restrict__ systemMatrix_dev,
                                    const double*  __restrict__ vVect_dev, 
                                    const long*  __restrict__ matrixIndexAstro_dev, 
                                    const long mapNoss, 
                                    const long offLocalAstro, 
                                    const short  nAstroPSolved){
    long jstartAstro = 0;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (ix < mapNoss) {
        double sum = 0.0f;
        jstartAstro = matrixIndexAstro_dev[ix] - offLocalAstro;
        #pragma unroll
        for(auto jx = 0; jx <   nAstroPSolved; jx++) {
           sum = sum + systemMatrix_dev[ix*nAstroPSolved + jx] * vVect_dev[jstartAstro+jx];
        }
        knownTerms_dev[ix] = knownTerms_dev[ix] + sum;

        ix+=gridDim.x*blockDim.x;
    }
}


// Attitude part
__global__ void aprod1_Kernel_att_AttAxis(double*  __restrict__ knownTerms_dev, 
                                        const double*  __restrict__ systemMatrix_dev, 
                                        const double*  __restrict__ vVect_dev, 
                                        const long*  __restrict__ matrixIndexAtt_dev, 
                                        const long  nAttP, 
                                        const long mapNoss, 
                                        const long nDegFreedomAtt, 
                                        const int offLocalAtt, 
                                        const short nAttParAxis)
{
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    while (ix < mapNoss) {
        double sum = 0.0f;
        const long miValAtt=matrixIndexAtt_dev[ix];
        long jstartAtt_0 = miValAtt + offLocalAtt; 
        #pragma unroll
        for(auto inpax = 0;inpax<nAttParAxis;++inpax)
            sum = sum + systemMatrix_dev[ix * nAttP + inpax] * vVect_dev[jstartAtt_0 + inpax];
        jstartAtt_0 += nDegFreedomAtt;
        #pragma unroll
        for(auto inpax = 0;inpax<nAttParAxis;inpax++)
            sum = sum + systemMatrix_dev[ix * nAttP + nAttParAxis + inpax ] * vVect_dev[jstartAtt_0 + inpax];
        jstartAtt_0 += nDegFreedomAtt;
        #pragma unroll
        for(auto inpax = 0;inpax<nAttParAxis;inpax++)
            sum = sum + systemMatrix_dev[ix * nAttP + nAttParAxis + nAttParAxis + inpax] * vVect_dev[jstartAtt_0 + inpax];

        knownTerms_dev[ix] = knownTerms_dev[ix] + sum;
        ix+=gridDim.x*blockDim.x;
  }
}



// Instrumental part
__global__ void aprod1_Kernel_instr(double*  __restrict__ knownTerms_dev, 
                                    const double*  __restrict__ systemMatrix_dev, 
                                    const double*  __restrict__ vVect_dev, 
                                    const int*  __restrict__ instrCol_dev, 
                                    const long mapNoss, 
                                    const long offLocalInstr, 
                                    const short  nInstrPSolved)
{
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    while (ix < mapNoss) {
        double sum = 0.0f;
        const long iiVal=ix*nInstrPSolved;
        long ixInstr = 0;
        #pragma unroll
        for(auto inInstr=0;inInstr<nInstrPSolved;inInstr++){
            ixInstr=offLocalInstr+instrCol_dev[iiVal+inInstr];
            sum += systemMatrix_dev[ix * nInstrPSolved + inInstr]*vVect_dev[ixInstr];
        }
        knownTerms_dev[ix] = knownTerms_dev[ix] + sum;
        ix+=gridDim.x*blockDim.x;
    }
}



// Global part
__global__ void aprod1_Kernel_glob(double*  __restrict__ knownTerms_dev, 
                                    const double*  __restrict__ systemMatrix_dev, 
                                    const double*  __restrict__ vVect_dev, 
                                    const long offLocalGlob, 
                                    const long mapNoss, 
                                    const short nGlobP)
{
    long ix = blockIdx.x * blockDim.x + threadIdx.x;   
    while (ix < mapNoss) {
        double sum = 0.0;
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
__global__ void aprod1_Kernel_ExtConstr(double*  __restrict__ knownTerms_dev,
                                        const double*  __restrict__ systemMatrix_dev, 
                                        const double*  __restrict__ vVect_dev, 
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


    for (int iexc = 0; iexc < nEqExtConstr; iexc++) {
        double sum = 0.0;
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
__global__ void aprod1_Kernel_BarConstr(double*  __restrict__ knownTerms_dev,
                                        const double*  __restrict__ systemMatrix_dev, 
                                        const double*  __restrict__ vVect_dev,
                                        const int nOfElextObs, 
                                        const int nOfElBarObs, 
                                        const int nEqExtConstr, 
                                        const long mapNoss, 
                                        const int nEqBarConstr, 
                                        const int numOfBarStar, 
                                        const short nAstroPSolved){
    long offBarConstrIx;
    long ktIx = mapNoss + nEqExtConstr;    
    long j3 = blockIdx.x * blockDim.x + threadIdx.x;   

    for(int iexc=0;iexc<nEqBarConstr;iexc++ ){
        double sum=0.0f;
        offBarConstrIx=iexc*nOfElBarObs;
        if (j3 < numOfBarStar*nAstroPSolved)
            sum = sum + systemMatrix_dev[offBarConstrIx+j3]*vVect_dev[j3];

        atomicAdd(&knownTerms_dev[ktIx+iexc],sum);
    }
}



/// InstrConstr
/// Mode 1 InstrConstr
__global__ void aprod1_Kernel_InstrConstr(double*  __restrict__ knownTerms_dev,
                                        const double*  __restrict__ systemMatrix_dev, 
                                        const double*  __restrict__ vVect_dev,
                                        const int*  __restrict__ instrConstrIlung_dev, 
                                        const int*  __restrict__ instrCol_dev, 
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
    const int i1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int i1_Aux = myid + i1*nproc;
    long offSetInstrConstr1=VrIdAstroPDimMax*nAstroPSolved+nDegFreedomAtt*nAttAxes;
    long offSetInstrInc=nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr;
    long offvV=0;
    int offSetInstr=0;
    int vVix=0;
    
    
    if(i1_Aux < nOfInstrConstr){
        double sum = 0.0;
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
                                    const long*  __restrict__ matrixIndexAstro_dev, 
                                    const long*  __restrict__ startend_dev,  
                                    const long offLocalAstro, 
                                    const long mapNoss, 
                                    const short nAstroPSolved){   
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    while(ix < mapNoss)
    {
        long tid=matrixIndexAstro_dev[startend_dev[ix]];
        for(long i=startend_dev[ix]; i<startend_dev[ix+1]; ++i){
            #pragma unroll
            for (auto jx = 0; jx < nAstroPSolved; ++jx)
                vVect_dev[tid - offLocalAstro + jx]+= systemMatrix_dev[i*nAstroPSolved + jx] * knownTerms_dev[i];
        }
        ix+=gridDim.x*blockDim.x;
   }
}




__global__ void aprod2_Kernel_att_AttAxis(double * __restrict__ vVect_dev, 
                                            const double * __restrict__ systemMatrix_dev, 
                                            const double * __restrict__ knownTerms_dev, 
                                            const long*  __restrict__ matrixIndexAtt_dev, 
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
        for (auto inpax = 0; inpax < nAttParAxis; inpax++)
            atomicAdd(&vVect_dev[jstartAtt + inpax],systemMatrix_dev[ix * nAttP + inpax] * knownTerms_dev[ix]);
        jstartAtt +=nDegFreedomAtt;
        #pragma unroll
        for (auto inpax = 0; inpax < nAttParAxis; inpax++)
            atomicAdd(&vVect_dev[jstartAtt + inpax],systemMatrix_dev[ix * nAttP + nAttParAxis + inpax] * knownTerms_dev[ix]);
        jstartAtt +=nDegFreedomAtt;
        #pragma unroll
        for (auto inpax = 0; inpax < nAttParAxis; inpax++)
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
        #pragma unroll
        for (auto inInstr = 0; inInstr < nInstrPSolved; inInstr++)
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
    shArr[threadIdx.x] = 0.0f;
    for (long ix = gthIdx; ix < mapNoss; ix += gridSize)
        shArr[threadIdx.x] += systemMatrix_dev[ix * nGlobP + inGlob] * knownTerms_dev[ix];
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { 
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
    double sum = 0.0f;
    for (long i = gthIdx; i < arraySize; i += gridSize)
        sum += gArr[i];
    __shared__ double shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { 
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
__global__ void aprod2_Kernel_ExtConstr(double*  __restrict__ vVect_dev, 
                                        const double*  __restrict__ systemMatrix_dev, 
                                        const double*  __restrict__ knownTerms_dev, 
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

    for(int ix = 0; ix < nEqExtConstr; ix++ ){  
        yi = knownTerms_dev[mapNoss + ix];
        if (i < numOfExtStar) {
            off3 = i*nAstroPSolved;
            off2 = ix*nOfElextObs + off3;
            for(short j2 = 0; j2 < nAstroPSolved; j2++){
                vVect_dev[j2+off3] += systemMatrix_dev[off2+j2]*yi;
            }
        }
    } 

    for(int ix=0;ix<nEqExtConstr;ix++ ){  
        yi = knownTerms_dev[mapNoss + ix];
        offExtAttConstrEq =  ix*nOfElextObs + numOfExtStar*nAstroPSolved; 
        for(short nax = 0; nax < nAttAxes; nax++){
            offExtUnk = off1 + nax*nDegFreedomAtt; 
            off2=offExtAttConstrEq+nax*numOfExtAttCol;

            if (i < numOfExtAttCol) {
                vVect_dev[offExtUnk+i] = vVect_dev[offExtUnk+i] + systemMatrix_dev[off2+i]*yi;
            }
        }
    }
}



__global__ void aprod2_Kernel_BarConstr (double*  __restrict__ vVect_dev,
                                        const double*  __restrict__ systemMatrix_dev, 
                                        const double*  __restrict__ knownTerms_dev, 
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
    
    for(int ix=0;ix<nEqBarConstr;ix++ ){  
        yi = knownTerms_dev[mapNoss+nEqExtConstr+ix];
        offBarStarConstrEq = nEqExtConstr*nOfElextObs+ix*nOfElBarObs;
        if (yx < numOfBarStar) {
            for(short j2=0;j2<nAstroPSolved;j2++)
                vVect_dev[j2+yx*nAstroPSolved] = vVect_dev[j2+yx*nAstroPSolved] + systemMatrix_dev[offBarStarConstrEq+yx*nAstroPSolved+j2]*yi;
        }
    } 
}




__global__ void aprod2_Kernel_InstrConstr(double*  __restrict__ vVect_dev,
                                        const double*  __restrict__ systemMatrix_dev, 
                                        const double*  __restrict__ knownTerms_dev, 
                                        const int*  __restrict__ instrConstrIlung_dev, 
                                        const int*  __restrict__ instrCol_dev, 
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

    checkHip( hipMemcpy(startend_dev, startend, sizeof(long)*(nnz2+1), hipMemcpyHostToDevice));

    free(startend);
    return nnz2;
}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------
// LSQR
// ---------------------------------------------------------------------
void lsqr(
          long int m,
          long int n,
          double damp,
          double *knownTerms,     
          double *vVect,     
          double *wVect,     
          double *xSolution,     
          double *standardError,    
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
          long *matrixIndexAstro, 
          long *matrixIndexAtt, 
          int *instrCol,
          int *instrConstrIlung,
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

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: TIME 1
    double communicationtime=ZERO, timeiter=ZERO, starttime=ZERO, GlobalTime=ZERO, TimeSetInit=ZERO, LoopTime=ZERO;
    MPI_Barrier(MPI_COMM_WORLD);
    TimeSetInit=MPI_Wtime();
    GlobalTime=MPI_Wtime();

    const int myid=comlsqr.myid;
    const long mapNoss=static_cast<long>(comlsqr.mapNoss[myid]);

    int deviceCount = 0;
    checkHip( hipGetDeviceCount(&deviceCount) );
    checkHip( hipSetDevice(myid % deviceCount) );
    int deviceNum = 0;
    checkHip( hipGetDevice(&deviceNum) );


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
    double startCycleTime,endCycleTime,totTime;
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

    hipStream_t stream;
    hipStreamCreate(&stream);

    hipStream_t streamAprod2_0;
    hipStream_t streamAprod2_1;
    hipStream_t streamAprod2_2;
    hipStream_t streamAprod2_3;
    hipStream_t streamAprod2_4;
    hipStream_t streamAprod2_5;
    hipStream_t streamAprod2_6;
    hipStream_t streamAprod2_7;
    hipStreamCreate(&streamAprod2_0);
    hipStreamCreate(&streamAprod2_1);
    hipStreamCreate(&streamAprod2_2);
    hipStreamCreate(&streamAprod2_3);
    hipStreamCreate(&streamAprod2_4);
    hipStreamCreate(&streamAprod2_5);
    hipStreamCreate(&streamAprod2_6);
    hipStreamCreate(&streamAprod2_7);


    if(nAstroPSolved){
        checkHip(hipMalloc((void**)&sysmatAstro_dev, mapNoss*nAstroPSolved*sizeof(double)) );
        checkHip(hipMemcpyAsync(sysmatAstro_dev, sysmatAstro, mapNoss*nAstroPSolved*sizeof(double), hipMemcpyHostToDevice,stream) );
    }
    if(nAttP){
        checkHip(hipMalloc((void**)&sysmatAtt_dev, mapNoss*nAttP*sizeof(double)) );
        checkHip(hipMemcpyAsync(sysmatAtt_dev, sysmatAtt, mapNoss*nAttP*sizeof(double), hipMemcpyHostToDevice,stream) );
    }
    if(nInstrPSolved){
        checkHip(hipMalloc((void**)&sysmatInstr_dev, mapNoss*nInstrPSolved*sizeof(double)) );
        checkHip(hipMemcpyAsync(sysmatInstr_dev, sysmatInstr, mapNoss*nInstrPSolved*sizeof(double), hipMemcpyHostToDevice,stream) );
    }
    if(nGlobP){
        checkHip(hipMalloc((void**)&sysmatGloB_dev, mapNoss*nGlobP*sizeof(double)) );
        checkHip(hipMemcpyAsync(sysmatGloB_dev, sysmatGloB, mapNoss*nGlobP*sizeof(double), hipMemcpyHostToDevice,stream) );
    }
    if(nTotConstraints){
        checkHip(hipMalloc((void**)&sysmatConstr_dev, (nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr+nElemIC)*sizeof(double)) );
        checkHip(hipMemcpyAsync(sysmatConstr_dev, sysmatConstr, (nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr+nElemIC)*sizeof(double), hipMemcpyHostToDevice,stream) );
    }


    // New list    
    long nnz=1;
    for(long i=0; i<mapNoss-1; i++){
        if(matrixIndexAstro[i]!=matrixIndexAstro[i+1]){
            nnz++;
        }
    }


    checkHip(hipMalloc((void**)&startend_dev, sizeof(long)*(nnz+1)));  
    nnz=create_startend_gpulist(matrixIndexAstro,startend_dev,mapNoss,nnz);

    checkHip( hipMalloc((void**)&matrixIndexAstro_dev, mapNoss*sizeof(long)));
    checkHip( hipMalloc((void**)&matrixIndexAtt_dev, mapNoss*sizeof(long)));     
    checkHip( hipMemcpyAsync(matrixIndexAstro_dev, matrixIndexAstro, mapNoss*sizeof(long), hipMemcpyHostToDevice,stream));
    checkHip( hipMemcpyAsync(matrixIndexAtt_dev, matrixIndexAtt, mapNoss*sizeof(long), hipMemcpyHostToDevice,stream));
    //--------------------------------------------------------------------------------------------------------

    checkHip(hipMalloc((void**)&vVect_dev, nunkSplit*sizeof(double)) );
    checkHip(hipMalloc((void**)&knownTerms_dev, nElemKnownTerms*sizeof(double)) );
    checkHip(hipMalloc((void**)&wVect_dev, nunkSplit*sizeof(double)) );
    checkHip(hipMalloc((void**)&kAuxcopy_dev, (nEqExtConstr+nEqBarConstr+nOfInstrConstr)*sizeof(double)) );
    checkHip(hipMalloc((void**)&vAuxVect_dev, localAstroMax*sizeof(double)) );
    checkHip(hipMalloc((void**)&instrCol_dev,(nInstrPSolved*mapNoss+nElemIC)*sizeof(int)) );  // nobs -> mapNoss.
    checkHip(hipMalloc((void**)&instrConstrIlung_dev,nOfInstrConstr*sizeof(int)) );
    checkHip(hipMalloc((void**)&xSolution_dev, nunkSplit*sizeof(double)) );
    checkHip(hipMalloc((void**)&standardError_dev, nunkSplit*sizeof(double)) );




    for(auto i=0; i<nunkSplit;++i){
        xSolution[i]=0.0;
        standardError[i]=0.0;
    }

    checkHip (hipMemcpyAsync(xSolution_dev, xSolution, nunkSplit*sizeof(double), hipMemcpyHostToDevice,stream) );
    checkHip (hipMemcpyAsync(standardError_dev, standardError, nunkSplit*sizeof(double), hipMemcpyHostToDevice,stream) );


    //  Copies H2D:
    checkHip(hipMemcpyAsync(instrCol_dev, instrCol, (nInstrPSolved*mapNoss+nElemIC)*sizeof(int), hipMemcpyHostToDevice,stream) );  
    checkHip(hipMemcpyAsync(instrConstrIlung_dev, instrConstrIlung, nOfInstrConstr*sizeof(int), hipMemcpyHostToDevice,stream));
    checkHip(hipMemcpyAsync(knownTerms_dev, knownTerms, nElemKnownTerms*sizeof(double), hipMemcpyHostToDevice,stream));
    checkHip(hipMalloc((void**)&dev_vVect_glob_sum, sizeof(double)*gridSize));
    checkHip(hipMalloc((void**)&dev_max_knownTerms, sizeof(double)*gridSize));
    checkHip(hipMalloc((void**)&dev_ssq_knownTerms, sizeof(double)*gridSize));
    checkHip(hipMalloc((void**)&dev_max_vVect, sizeof(double)*gridSize));
    checkHip(hipMalloc((void**)&dev_ssq_vVect, sizeof(double)*gridSize));

    

    double* dknorm_vec;
    hipMalloc((void**)&dknorm_vec,sizeof(double));


    //  Grid topologies:
    dim3 blockDim(TILE_WIDTH,1,1);
    dim3 gridDim_aprod1((mapNoss - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_aprod1_Plus_Constr((mapNoss + nEqExtConstr + nEqBarConstr + nOfInstrConstr - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_vAuxVect_Kernel((localAstroMax - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_vVect_Put_To_Zero_Kernel((nunkSplit - 1)/TILE_WIDTH + 1,1,1);
    dim3 gridDim_nunk((nunkSplit - 1)/TILE_WIDTH + 1,1,1);
    // dim3 gridDim_aprod2((mapNoss - 1)/TILE_WIDTH + 1,1,1);

    //old
    // dim3 gridDim_aprod2Astro((mapNoss - 1)/ThreadsXBlockAprod2Astro + 1,1,1);
    // dim3 gridDim_aprod2Att((mapNoss - 1)/ThreadsXBlockAprod2Att + 1,1,1);
    // dim3 gridDim_aprod2Instr((mapNoss - 1)/ThreadsXBlockAprod2Instr + 1,1,1);


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
    

////////////////////////////////////////////// 
other=(long)nAttParam + nInstrParam + nGlobalParam; 
comlsqr.itn=itn;
totTime=comlsqr.totSec;

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
    
checkHip (hipMemcpyAsync(vVect_dev, vVect, nunkSplit*sizeof(double), hipMemcpyHostToDevice,stream) );
    
dload( nunkSplit, 0.0, xSolution );

if ( wantse )   dload( nunkSplit, 0.0, standardError );

alpha  =   ZERO;
    
maxCommMultiBlock_double<<<gridSize, blockSize>>>(knownTerms_dev, dev_max_knownTerms, nElemKnownTerms);
maxCommMultiBlock_double<<<1, blockSize>>>(dev_max_knownTerms, dev_max_knownTerms, gridSize);
checkHip (hipMemcpyAsync(&max_knownTerms, dev_max_knownTerms, sizeof(double), hipMemcpyDeviceToHost,stream) );

double betaLoc, betaLoc2;
if (myid == 0) {
    sumCommMultiBlock_double<<<gridSize, blockSize>>>(knownTerms_dev,dev_ssq_knownTerms,max_knownTerms,nElemKnownTerms);
    realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_knownTerms,dev_ssq_knownTerms,gridSize);
    checkHip(hipMemcpyAsync(&ssq_knownTerms, dev_ssq_knownTerms, sizeof(double), hipMemcpyDeviceToHost,stream) );
    betaLoc = max_knownTerms*sqrt(ssq_knownTerms);
} else {
    sumCommMultiBlock_double<<<gridSize, blockSize>>>(knownTerms_dev,dev_ssq_knownTerms,max_knownTerms,mapNoss);
    realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_knownTerms,dev_ssq_knownTerms,gridSize);
    checkHip(hipMemcpyAsync(&ssq_knownTerms, dev_ssq_knownTerms, sizeof(double), hipMemcpyDeviceToHost,stream) );
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

    dscal<<<BlockXGrid,ThreadsXBlock>>>(knownTerms_dev, 1.0/beta,nElemKnownTerms, 1.0);
	if(myid!=0)
	{
        vVect_Put_To_Zero_Kernel<<<gridDim_vVect_Put_To_Zero_Kernel,TILE_WIDTH>>>(vVect_dev,localAstroMax,nunkSplit);
	}
        
    //APROD2 CALL BEFORE LSQR
    hipDeviceSynchronize();
    if(nAstroPSolved) aprod2_Kernel_astro<<<BlockXGridAprod2Astro,ThreadsXBlockAprod2Astro,0,streamAprod2_0>>>(vVect_dev, sysmatAstro_dev, knownTerms_dev, matrixIndexAstro_dev, startend_dev, offLocalAstro, nnz, nAstroPSolved);
    if(nAttP) aprod2_Kernel_att_AttAxis<<<BlockXGrid,ThreadsXBlock,0,streamAprod2_1>>>(vVect_dev, sysmatAtt_dev, knownTerms_dev, matrixIndexAtt_dev, nAttP, nDegFreedomAtt, offLocalAtt, mapNoss, nAstroPSolved, nAttParAxis);
    if(nInstrPSolved) aprod2_Kernel_instr<<<BlockXGrid,ThreadsXBlock,0,streamAprod2_2>>>(vVect_dev, sysmatInstr_dev, knownTerms_dev, instrCol_dev, offLocalInstr, mapNoss, nInstrPSolved);
    for (short inGlob = 0; inGlob < nGlobP; inGlob++)
    {
        sumCommMultiBlock_double_aprod2_Kernel_glob<<<gridSize, blockSize,0,streamAprod2_3>>>(dev_vVect_glob_sum, sysmatGloB_dev, knownTerms_dev, vVect_dev, nGlobP, mapNoss, offLocalGlob, inGlob);
        realsumCommMultiBlock_double_aprod2_Kernel_glob<<<1, blockSize,0,streamAprod2_4>>>(vVect_dev,dev_vVect_glob_sum, gridSize, offLocalGlob, inGlob);
    }
    hipDeviceSynchronize();
    //  CONSTRAINTS OF APROD MODE 2:
    if(nEqExtConstr) aprod2_Kernel_ExtConstr<<<gridDim_aprod2_ExtConstr, TILE_WIDTH,0,streamAprod2_5>>>(vVect_dev,sysmatInstr_dev,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAttAxes,nAstroPSolved);
    if(nEqBarConstr) aprod2_Kernel_BarConstr<<<gridDim_aprod2_BarConstr, TILE_WIDTH,0,streamAprod2_6>>>(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
    if(nOfInstrConstr) aprod2_Kernel_InstrConstr<<<gridDim_aprod2_InstrConstr, TILE_WIDTH,0,streamAprod2_7>>>(vVect_dev,sysmatConstr_dev,knownTerms_dev,instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,nDegFreedomAtt,mapNoss,nEqExtConstr,nEqBarConstr,nOfElextObs,nOfElBarObs,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);

    // hipDeviceSynchronize();
    // if(nAstroPSolved) aprod2_Kernel_astro<<<BlockXGridAprod2Astro,ThreadsXBlockAprod2Astro,0,streamAprod2_0>>>(vVect_dev, sysmatAstro_dev, knownTerms_dev, matrixIndexAstro_dev, startend_dev, offLocalAstro, nnz, nAstroPSolved);
    // if(nAttP) aprod2_Kernel_att_AttAxis<<<BlockXGridAprod2Att,ThreadsXBlockAprod2Att,0,streamAprod2_1>>>(vVect_dev, sysmatAtt_dev, knownTerms_dev, matrixIndexAtt_dev, nAttP, nDegFreedomAtt, offLocalAtt, mapNoss, nAstroPSolved, nAttParAxis);
    // if(nInstrPSolved) aprod2_Kernel_instr<<<BlockXGridAprod2Instr,ThreadsXBlockAprod2Instr,0,streamAprod2_2>>>(vVect_dev, sysmatInstr_dev, knownTerms_dev, instrCol_dev, offLocalInstr, mapNoss, nInstrPSolved);
    // for (short inGlob = 0; inGlob < nGlobP; inGlob++)
    // {
    //     sumCommMultiBlock_double_aprod2_Kernel_glob<<<gridSize, blockSize,0,streamAprod2_3>>>(dev_vVect_glob_sum, sysmatGloB_dev, knownTerms_dev, vVect_dev, nGlobP, mapNoss, offLocalGlob, inGlob);
    //     realsumCommMultiBlock_double_aprod2_Kernel_glob<<<1, blockSize,0,streamAprod2_4>>>(vVect_dev,dev_vVect_glob_sum, gridSize, offLocalGlob, inGlob);
    // }
    // hipDeviceSynchronize();
    // //  CONSTRAINTS OF APROD MODE 2:
    // if(nEqExtConstr) aprod2_Kernel_ExtConstr<<<gridDim_aprod2_ExtConstr, TILE_WIDTH,0,streamAprod2_5>>>(vVect_dev,sysmatInstr_dev,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAttAxes,nAstroPSolved);
    // if(nEqBarConstr) aprod2_Kernel_BarConstr<<<gridDim_aprod2_BarConstr, TILE_WIDTH,0,streamAprod2_6>>>(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
    // if(nOfInstrConstr) aprod2_Kernel_InstrConstr<<<gridDim_aprod2_InstrConstr, TILE_WIDTH,0,streamAprod2_7>>>(vVect_dev,sysmatConstr_dev,knownTerms_dev,instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,nDegFreedomAtt,mapNoss,nEqExtConstr,nEqBarConstr,nOfElextObs,nOfElBarObs,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);



  

    checkHip ( hipMemcpyAsync(vVect, vVect_dev, nunkSplit*sizeof(double), hipMemcpyDeviceToHost,streamAprod2_7) );

    /* ~~~~~~~~~~~~~~ */
    hipDeviceSynchronize();
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

    
    checkHip (hipMemcpyAsync(vVect_dev, vVect, nunkSplit*sizeof(double), hipMemcpyHostToDevice,stream) );

    nAstroElements=comlsqr.mapStar[myid][1]-comlsqr.mapStar[myid][0] + 1;
 	if(myid<nproc-1)
 	{
 		nAstroElements=comlsqr.mapStar[myid][1]-comlsqr.mapStar[myid][0] +1;
 		if(comlsqr.mapStar[myid][1]==comlsqr.mapStar[myid+1][0]) nAstroElements--;
 	}

    // reset internal state
    maxCommMultiBlock_double<<<gridSize, blockSize>>>(vVect_dev, dev_max_vVect, nunkSplit);
    maxCommMultiBlock_double<<<1, blockSize>>>(dev_max_vVect, dev_max_vVect, gridSize);
    
        
    hipMemcpyAsync(&max_vVect, dev_max_vVect, sizeof(double), hipMemcpyDeviceToHost,stream);
 
    double alphaLoc=0.0;
    
    sumCommMultiBlock_double<<<gridSize, blockSize>>>(vVect_dev, dev_ssq_vVect, max_vVect, nAstroElements*nAstroPSolved);
    realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_vVect, dev_ssq_vVect, gridSize);
    hipMemcpyAsync(&ssq_vVect, dev_ssq_vVect, sizeof(double), hipMemcpyDeviceToHost,stream);
    alphaLoc = max_vVect*sqrt(ssq_vVect);
     
    alphaLoc2=alphaLoc*alphaLoc;
	if(myid==0) {
        double alphaOther2 = alphaLoc2;
        sumCommMultiBlock_double<<<gridSize, blockSize>>>(&vVect_dev[localAstroMax], dev_ssq_vVect, max_vVect, nunkSplit - localAstroMax);
        realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_vVect, dev_ssq_vVect, gridSize);
        checkHip (hipMemcpyAsync(&ssq_vVect, dev_ssq_vVect, sizeof(double), hipMemcpyDeviceToHost,stream) );
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
        dscal<<<BlockXGrid,ThreadsXBlock>>>(vVect_dev, 1/alpha, nunkSplit, 1.0);
        cblas_dcopy_kernel<<<gridDim_nunk,TILE_WIDTH>>>(nunkSplit,vVect_dev,wVect_dev);
    }
    
    checkHip (hipMemcpyAsync(vVect, vVect_dev, nunkSplit*sizeof(double), hipMemcpyDeviceToHost,stream) );
    checkHip (hipMemcpyAsync(wVect, wVect_dev, nunkSplit*sizeof(double), hipMemcpyDeviceToHost,stream) );
    checkHip (hipMemcpyAsync(knownTerms, knownTerms_dev, nElemKnownTerms*sizeof(double), hipMemcpyDeviceToHost,stream) );


    arnorm  = alpha * beta;

    if (arnorm == ZERO){
        if (damped  &&  istop == 2) istop = 3;

        checkHip(hipFree(vVect_dev));
        checkHip(hipFree(wVect_dev));
        checkHip(hipFree(knownTerms_dev));
        checkHip(hipFree(kAuxcopy_dev));
        checkHip(hipFree(vAuxVect_dev));
        checkHip(hipFree(instrCol_dev));
        checkHip(hipFree(instrConstrIlung_dev));
        checkHip(hipFree(dev_vVect_glob_sum));
        checkHip(hipFree(dev_max_knownTerms)); 
        checkHip(hipFree(dev_ssq_knownTerms)); 
        checkHip(hipFree(dev_max_vVect)); 
        checkHip(hipFree(dev_ssq_vVect)); 
        checkHip(hipFree(matrixIndexAstro_dev));
        checkHip(hipFree(startend_dev));

        checkHip(hipFree(sysmatAstro_dev));
        checkHip(hipFree(sysmatAtt_dev));
        checkHip(hipFree(sysmatInstr_dev));
        checkHip(hipFree(sysmatGloB_dev));
        checkHip(hipFree(sysmatConstr_dev));

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

    
    checkHip (hipMemcpyAsync(knownTerms_dev, knownTerms, nElemKnownTerms*sizeof(double), hipMemcpyHostToDevice,stream) );
    checkHip (hipMemcpyAsync(vVect_dev, vVect, nunkSplit*sizeof(double), hipMemcpyHostToDevice,stream) );
    checkHip (hipMemcpyAsync(wVect_dev, wVect, nunkSplit*sizeof(double), hipMemcpyHostToDevice,stream) );
    
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
        dscal<<<BlockXGrid,ThreadsXBlock,0,stream>>>(knownTerms_dev, alpha, nElemKnownTerms, -1.0);
        kAuxcopy_Kernel<<<gridDim_kAuxcopy_Kernel, TILE_WIDTH,0,stream>>>(knownTerms_dev,kAuxcopy_dev,mapNoss,nEqExtConstr+nEqBarConstr+nOfInstrConstr);
        //{ // CONTEXT MODE 1//////////////////////////////////// APROD MODE 1
        if(nAstroPSolved) aprod1_Kernel_astro<<<gridDim_aprod1,TILE_WIDTH,0,stream>>>(knownTerms_dev, sysmatAstro_dev, vVect_dev, matrixIndexAstro_dev, mapNoss, offLocalAstro, nAstroPSolved);
        if(nAttP) aprod1_Kernel_att_AttAxis<<<gridDim_aprod1,TILE_WIDTH,0,stream>>>(knownTerms_dev, sysmatAtt_dev, vVect_dev, matrixIndexAtt_dev, nAttP, mapNoss, nDegFreedomAtt, offLocalAtt, nAttParAxis);                
        if(nInstrPSolved) aprod1_Kernel_instr<<<gridDim_aprod1,TILE_WIDTH,0,stream>>>(knownTerms_dev, sysmatInstr_dev, vVect_dev, instrCol_dev, mapNoss, offLocalInstr, nInstrPSolved);
        if(nGlobP) aprod1_Kernel_glob<<<gridDim_aprod1,TILE_WIDTH,0,stream>>>(knownTerms_dev, sysmatGloB_dev , vVect_dev, offLocalGlob, mapNoss, nGlobP);
        //        CONSTRAINTS APROD MODE 1        
        if(nEqExtConstr) aprod1_Kernel_ExtConstr<<<gridDim_aprod1_ExtConstr,TILE_WIDTH,0,stream>>>(knownTerms_dev,sysmatConstr_dev,vVect_dev,VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,startingAttColExtConstr,nEqExtConstr,nOfElextObs,numOfExtStar, numOfExtAttCol,nAstroPSolved,nAttAxes);
        if(nEqBarConstr) aprod1_Kernel_BarConstr<<<gridDim_aprod1_BarConstr,TILE_WIDTH,0,stream>>>(knownTerms_dev,sysmatConstr_dev,vVect_dev, nOfElextObs,nOfElBarObs,nEqExtConstr,mapNoss,nEqBarConstr,numOfBarStar,nAstroPSolved);
        if(nOfInstrConstr) aprod1_Kernel_InstrConstr<<<gridDim_aprod1_InstrConstr,TILE_WIDTH,0,stream>>>(knownTerms_dev,sysmatConstr_dev,vVect_dev,instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,mapNoss, nDegFreedomAtt,nOfElextObs,nEqExtConstr,nOfElBarObs,nEqBarConstr,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);
        //}// END CONTEXT MODE 1
        checkHip (hipMemcpyAsync(&knownTerms[mapNoss], &knownTerms_dev[mapNoss], (nEqExtConstr+nEqBarConstr+nOfInstrConstr)*sizeof(double), hipMemcpyDeviceToHost,stream) );
        hipDeviceSynchronize();
        //------------------------------------------------------------------------------------------------
        starttime=MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE,&knownTerms[mapNoss],nEqExtConstr+nEqBarConstr+nOfInstrConstr,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //------------------------------------------------------------------------------------------------
        checkHip(hipMemcpyAsync(&knownTerms_dev[mapNoss], &knownTerms[mapNoss], (nEqExtConstr+nEqBarConstr+nOfInstrConstr)*sizeof(double), hipMemcpyHostToDevice,stream) );
        kauxsum<<<gridDim_kAuxcopy_Kernel,TILE_WIDTH,0,stream>>>(&knownTerms_dev[mapNoss],kAuxcopy_dev,nEqExtConstr+nEqBarConstr+nOfInstrConstr);
        maxCommMultiBlock_double<<<gridSize, blockSize,0,stream>>>(knownTerms_dev, dev_max_knownTerms, nElemKnownTerms);
        maxCommMultiBlock_double<<<1, blockSize,0,stream>>>(dev_max_knownTerms, dev_max_knownTerms, gridSize);
        hipMemcpyAsync(&max_knownTerms, dev_max_knownTerms, sizeof(double), hipMemcpyDeviceToHost,stream);
        
        if(myid==0)
        {
            sumCommMultiBlock_double<<<gridSize, blockSize>>>(knownTerms_dev, dev_ssq_knownTerms, max_knownTerms, mapNoss + nEqExtConstr + nEqBarConstr+nOfInstrConstr);
            realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_knownTerms, dev_ssq_knownTerms, gridSize);
            checkHip (hipMemcpyAsync(&ssq_knownTerms, dev_ssq_knownTerms, sizeof(double), hipMemcpyDeviceToHost,stream) );
            betaLoc = max_knownTerms*sqrt(ssq_knownTerms);
            betaLoc2 = betaLoc*betaLoc;
        }else{
            sumCommMultiBlock_double<<<gridSize, blockSize>>>(knownTerms_dev, dev_ssq_knownTerms, max_knownTerms, mapNoss);
            realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_knownTerms, dev_ssq_knownTerms, gridSize);
            checkHip (hipMemcpyAsync(&ssq_knownTerms, dev_ssq_knownTerms, sizeof(double), hipMemcpyDeviceToHost,stream) );
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
            dscal<<<BlockXGrid,ThreadsXBlock,0,stream>>>(knownTerms_dev, 1/beta,nElemKnownTerms, 1.0);
            dscal<<<BlockXGrid,ThreadsXBlock,0,stream>>>(vVect_dev, beta, nunkSplit, -1.0);
            vAuxVect_Kernel<<<gridDim_vAuxVect_Kernel,TILE_WIDTH,0,stream>>>(vVect_dev, vAuxVect_dev, localAstroMax);
            if (myid != 0) {
                vVect_Put_To_Zero_Kernel<<<gridDim_vVect_Put_To_Zero_Kernel,TILE_WIDTH,0,stream>>>(vVect_dev,localAstroMax,nunkSplit);
            }
            //{ // CONTEXT MODE 2 //////////////////////////////////// APROD MODE 2
            //APROD2 CALL BEFORE LSQR
            hipDeviceSynchronize();
            if(nAstroPSolved) aprod2_Kernel_astro<<<BlockXGridAprod2Astro,ThreadsXBlockAprod2Astro,0,streamAprod2_0>>>(vVect_dev, sysmatAstro_dev, knownTerms_dev, matrixIndexAstro_dev, startend_dev, offLocalAstro, nnz, nAstroPSolved);
            if(nAttP) aprod2_Kernel_att_AttAxis<<<BlockXGrid,ThreadsXBlock,0,streamAprod2_1>>>(vVect_dev, sysmatAtt_dev, knownTerms_dev, matrixIndexAtt_dev, nAttP, nDegFreedomAtt, offLocalAtt, mapNoss, nAstroPSolved, nAttParAxis);
            if(nInstrPSolved) aprod2_Kernel_instr<<<BlockXGrid,ThreadsXBlock,0,streamAprod2_2>>>(vVect_dev, sysmatInstr_dev, knownTerms_dev, instrCol_dev, offLocalInstr, mapNoss, nInstrPSolved);
            for (short inGlob = 0; inGlob < nGlobP; inGlob++)
            {
                sumCommMultiBlock_double_aprod2_Kernel_glob<<<gridSize, blockSize,0,streamAprod2_3>>>(dev_vVect_glob_sum, sysmatGloB_dev, knownTerms_dev, vVect_dev, nGlobP, mapNoss, offLocalGlob, inGlob);
                realsumCommMultiBlock_double_aprod2_Kernel_glob<<<1, blockSize,0,streamAprod2_4>>>(vVect_dev,dev_vVect_glob_sum, gridSize, offLocalGlob, inGlob);
            }
            hipDeviceSynchronize();
            //  CONSTRAINTS OF APROD MODE 2:
            if(nEqExtConstr) aprod2_Kernel_ExtConstr<<<gridDim_aprod2_ExtConstr, TILE_WIDTH,0,streamAprod2_5>>>(vVect_dev,sysmatInstr_dev,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAttAxes,nAstroPSolved);
            if(nEqBarConstr) aprod2_Kernel_BarConstr<<<gridDim_aprod2_BarConstr, TILE_WIDTH,0,streamAprod2_6>>>(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
            if(nOfInstrConstr) aprod2_Kernel_InstrConstr<<<gridDim_aprod2_InstrConstr, TILE_WIDTH,0,streamAprod2_7>>>(vVect_dev,sysmatConstr_dev,knownTerms_dev,instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,nDegFreedomAtt,mapNoss,nEqExtConstr,nEqBarConstr,nOfElextObs,nOfElBarObs,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);

            // hipDeviceSynchronize();
            // if(nAstroPSolved) aprod2_Kernel_astro<<<BlockXGridAprod2Astro,ThreadsXBlockAprod2Astro,0,streamAprod2_0>>>(vVect_dev, sysmatAstro_dev, knownTerms_dev, matrixIndexAstro_dev, startend_dev, offLocalAstro, nnz, nAstroPSolved);
            // if(nAttP) aprod2_Kernel_att_AttAxis<<<BlockXGridAprod2Att,ThreadsXBlockAprod2Att,0,streamAprod2_1>>>(vVect_dev, sysmatAtt_dev, knownTerms_dev, matrixIndexAtt_dev, nAttP, nDegFreedomAtt, offLocalAtt, mapNoss, nAstroPSolved, nAttParAxis);
            // if(nInstrPSolved) aprod2_Kernel_instr<<<BlockXGridAprod2Instr,ThreadsXBlockAprod2Instr,0,streamAprod2_2>>>(vVect_dev, sysmatInstr_dev, knownTerms_dev, instrCol_dev, offLocalInstr, mapNoss, nInstrPSolved);
            // for (short inGlob = 0; inGlob < nGlobP; inGlob++)
            // {
            //     sumCommMultiBlock_double_aprod2_Kernel_glob<<<gridSize, blockSize,0,streamAprod2_3>>>(dev_vVect_glob_sum, sysmatGloB_dev, knownTerms_dev, vVect_dev, nGlobP, mapNoss, offLocalGlob, inGlob);
            //     realsumCommMultiBlock_double_aprod2_Kernel_glob<<<1, blockSize,0,streamAprod2_4>>>(vVect_dev,dev_vVect_glob_sum, gridSize, offLocalGlob, inGlob);
            // }
            // hipDeviceSynchronize();
            // //  CONSTRAINTS OF APROD MODE 2:
            // if(nEqExtConstr) aprod2_Kernel_ExtConstr<<<gridDim_aprod2_ExtConstr, TILE_WIDTH,0,streamAprod2_5>>>(vVect_dev,sysmatInstr_dev,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAttAxes,nAstroPSolved);
            // if(nEqBarConstr) aprod2_Kernel_BarConstr<<<gridDim_aprod2_BarConstr, TILE_WIDTH,0,streamAprod2_6>>>(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
            // if(nOfInstrConstr) aprod2_Kernel_InstrConstr<<<gridDim_aprod2_InstrConstr, TILE_WIDTH,0,streamAprod2_7>>>(vVect_dev,sysmatConstr_dev,knownTerms_dev,instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,nDegFreedomAtt,mapNoss,nEqExtConstr,nEqBarConstr,nOfElextObs,nOfElBarObs,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);





            checkHip ( hipMemcpyAsync(vVect, vVect_dev, nunkSplit*sizeof(double), hipMemcpyDeviceToHost,streamAprod2_7) );
            hipDeviceSynchronize();
            //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            //------------------------------------------------------------------------------------------------
            starttime=MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE,&vVect[localAstroMax], nAttParam+nInstrParam+nGlobalParam,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  
            communicationtime+=MPI_Wtime()-starttime;
            //------------------------------------------------------------------------------------------------
            //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::          
            if(nAstroPSolved) SumCirc2(vVect,comlsqr,&communicationtime);
            //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


            checkHip (hipMemcpyAsync(vVect_dev, vVect, nunkSplit*sizeof(double), hipMemcpyHostToDevice,stream) );
            vaux_sum<<<gridDim_vAuxVect_Kernel,TILE_WIDTH>>>(vVect_dev,vAuxVect_dev,localAstroMax);
            maxCommMultiBlock_double<<<gridSize, blockSize>>>(vVect_dev, dev_max_vVect, nunkSplit);
            maxCommMultiBlock_double<<<1, blockSize>>>(dev_max_vVect, dev_max_vVect, gridSize);
            checkHip (hipMemcpyAsync(&max_vVect, dev_max_vVect, sizeof(double), hipMemcpyDeviceToHost,stream) );
                        
            sumCommMultiBlock_double<<<gridSize, blockSize>>>(vVect_dev, dev_ssq_vVect, max_vVect, nAstroElements*nAstroPSolved);
            realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_vVect, dev_ssq_vVect, gridSize);
            checkHip (hipMemcpyAsync(&ssq_vVect, dev_ssq_vVect, sizeof(double), hipMemcpyDeviceToHost,stream) );
            
            double alphaLoc = 0.0;
            alphaLoc = max_vVect*sqrt(ssq_vVect);
            alphaLoc2=alphaLoc*alphaLoc;
    
            if(myid==0) {
                double alphaOther2 = alphaLoc2;
                sumCommMultiBlock_double<<<gridSize, blockSize>>>(&vVect_dev[localAstroMax], dev_ssq_vVect, max_vVect, nunkSplit - localAstroMax);
                realsumCommMultiBlock_double<<<1, blockSize>>>(dev_ssq_vVect, dev_ssq_vVect, gridSize);
                checkHip (hipMemcpyAsync(&ssq_vVect, dev_ssq_vVect, sizeof(double), hipMemcpyDeviceToHost,stream) );
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
                dscal<<<BlockXGrid,ThreadsXBlock>>>(vVect_dev, 1/alpha, nunkSplit, 1);
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


        dknorm_compute<<<gridSize,blockSize>>>(dknorm_vec,wVect_dev,0,nAstroElements*nAstroPSolved,t3);


        checkHip (hipMemcpy(&dknorm, dknorm_vec, sizeof(double), hipMemcpyDeviceToHost) );
        //------------------------------------------------------------------------------------------------
        starttime=MPI_Wtime();
 		MPI_Allreduce(MPI_IN_PLACE,&dknorm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //------------------------------------------------------------------------------------------------ 		
        checkHip (hipMemcpy(dknorm_vec,&dknorm, sizeof(double), hipMemcpyHostToDevice) );

        transform1<<<BlockXGrid,ThreadsXBlock>>>(xSolution_dev,wVect_dev,0,localAstro,t1);
        if(wantse)  transform2<<<BlockXGrid,ThreadsXBlock>>>(standardError_dev,wVect_dev,0,localAstro,t3);
        transform3<<<BlockXGrid,ThreadsXBlock>>>(wVect_dev,vVect_dev,0,localAstro,t2);

        transform1<<<BlockXGrid,ThreadsXBlock>>>(xSolution_dev,wVect_dev,localAstroMax,localAstroMax+other,t1);
        if(wantse)  transform2<<<BlockXGrid,ThreadsXBlock>>>(standardError_dev,wVect_dev,localAstroMax,localAstroMax+other,t3);
        transform3<<<BlockXGrid,ThreadsXBlock>>>(wVect_dev,vVect_dev,localAstroMax,localAstroMax+other,t2);

        dknorm_compute<<<gridSize,blockSize>>>(dknorm_vec,wVect_dev,localAstroMax,localAstroMax+other,t3);

        checkHip (hipMemcpy(&dknorm, dknorm_vec, sizeof(double), hipMemcpyDeviceToHost) );




        
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

    checkHip (hipMemcpy(xSolution,xSolution_dev, nunkSplit*sizeof(double), hipMemcpyDeviceToHost) );
    checkHip (hipMemcpy(standardError,standardError_dev, nunkSplit*sizeof(double), hipMemcpyDeviceToHost) );

    if ( wantse ) {
        t    =   ONE;
        if (m > n)     t = m - n;
        if ( damped )  t = m;
        t    =   rnorm / sqrt( t );
      
        for (long i = 0; i < nunkSplit; i++)
            standardError[i]  = t * sqrt( standardError[i] );
        
    }


    *istop_out  = istop;
    *itn_out    = itn;
    *anorm_out  = anorm;
    *acond_out  = acond;
    *rnorm_out  = rnorm;
    *arnorm_out = test2;
    *xnorm_out  = xnorm;


    checkHip(hipFree(xSolution_dev));
    checkHip(hipFree(standardError_dev));
    checkHip(hipFree(dknorm_vec));

    

    checkHip(hipFree(vVect_dev));
    checkHip(hipFree(wVect_dev));
    checkHip(hipFree(knownTerms_dev));
    checkHip(hipFree(kAuxcopy_dev));
    checkHip(hipFree(vAuxVect_dev));
    checkHip(hipFree(instrCol_dev));
    checkHip(hipFree(instrConstrIlung_dev));
    checkHip(hipFree(dev_vVect_glob_sum));
    checkHip(hipFree(dev_max_knownTerms)); 
    checkHip(hipFree(dev_ssq_knownTerms)); 
    checkHip(hipFree(dev_max_vVect)); 
    checkHip(hipFree(dev_ssq_vVect)); 
    checkHip(hipFree(matrixIndexAstro_dev));
    checkHip(hipFree(startend_dev));

    checkHip(hipFree(sysmatAstro_dev));
    checkHip(hipFree(sysmatAtt_dev));
    checkHip(hipFree(sysmatInstr_dev));
    checkHip(hipFree(sysmatGloB_dev));
    checkHip(hipFree(sysmatConstr_dev));

    hipStreamDestroy(stream);
    hipStreamDestroy(streamAprod2_0);
    hipStreamDestroy(streamAprod2_1);
    hipStreamDestroy(streamAprod2_2);
    hipStreamDestroy(streamAprod2_3);
    hipStreamDestroy(streamAprod2_4);
    hipStreamDestroy(streamAprod2_5);
    hipStreamDestroy(streamAprod2_6);
    hipStreamDestroy(streamAprod2_7);

    return;
}



