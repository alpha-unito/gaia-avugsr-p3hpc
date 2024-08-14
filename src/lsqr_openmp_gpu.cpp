/* 

OpenMP Version

*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#include "util.h"
#include <cassert>

#include <limits>
#include <sys/time.h>
#include <iostream>

#include <numeric>
#include <iostream>

#include <atomic>

static const int blockSize = 128;
static const int gridSize = 1024;


// AMD
#if defined(__AMD__)
    #ifndef AUTO_TUNING
        #if !defined(TEAMSAPROD2ASTRO)
        #define TEAMSAPROD2ASTRO 64
        #define THREADSAPROD2ASTRO 256
        #endif
        #if !defined(TEAMSAPROD2)
        #define TEAMSAPROD2 64
        #define THREADSAPROD2 256
        #endif
        #if !defined(TEAMSAPROD1)
        #define TEAMSAPROD1 256
        #define THREADSAPROD1 1024
        #endif
    #endif
#elif defined(__NVIDIA90__)
    // NVIDIA
    #ifndef AUTO_TUNING
        #if !defined(TEAMSAPROD2ASTRO)
        #define TEAMSAPROD2ASTRO 1024
        #define THREADSAPROD2ASTRO 64
        #endif
        #if !defined(TEAMSAPROD2)
        #define TEAMSAPROD2 1024
        #define THREADSAPROD2 32
        #endif
        #if !defined(TEAMSAPROD1)
        #define TEAMSAPROD1 4096
        #define THREADSAPROD1 32
        #endif
    #endif
#elif defined(__NVIDIA70__)
    // NVIDIA
    #ifndef AUTO_TUNING
        #if !defined(TEAMSAPROD2ASTRO)
        #define TEAMSAPROD2ASTRO 1024
        #define THREADSAPROD2ASTRO 64
        #endif
        #if !defined(TEAMSAPROD2)
        #define TEAMSAPROD2 1024
        #define THREADSAPROD2 32
        #endif
        #if !defined(TEAMSAPROD1)
        #define TEAMSAPROD1 1024
        #define THREADSAPROD1 32
        #endif
    #endif
#endif


#if _OPENMP
# include <omp.h>
#endif



#define ZERO   0.0
#define ONE    1.0
#define MONE    -1.0




int my_gpu=0; 



//-----------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------
inline double d2norm(const double& a, const double& b){

    double scale=std::fabs(a)+std::fabs(b);
    return (!scale) ? ZERO : scale * std::sqrt((a/scale)*(a/scale)+(b/scale)*(b/scale)); 

}

template<typename L>
inline void dload(double* x, const L n, const double alpha){

    #pragma omp parallel for 
    for(int i=0; i<n; ++i){
        x[i]=alpha;
    }
}

template<typename L>
inline void dscal(double* array, const double val, const L N, const double sign ){

    #pragma omp target teams distribute parallel for map(to:N) 
    for(int i=0; i<N; ++i){
        array[i]=sign*(array[i]*val);
    }
}


template<typename L>
inline double maxCommMultiBlock_double(double* gArr, const L arraySize){

    double max{ZERO};
    
    #pragma omp target teams distribute parallel for reduction(max : max) map(to:arraySize) 
    for(L i=0; i<arraySize; ++i){
        if(fabs(gArr[i])>max) max=fabs(gArr[i]);
    }

    return max;

}

template<typename L>
inline double sumCommMultiBlock_double(double* gArr, const L arraySize, const double max){

    const double d{1/max};
    double sum{ZERO};

    #pragma omp target teams distribute parallel for reduction(+:sum) map(to:arraySize) map(tofrom:max) 
    for(L i=0; i<arraySize; ++i){
        sum+=(gArr[i]*d)*(gArr[i]*d);
    }

    return sum;

}

template<typename L>
void vVect_Put_To_Zero_Kernel (double* vVect_dev, const L localAstroMax, const L nunkSplit)
{
    #pragma omp target teams distribute parallel for map(to:localAstroMax,nunkSplit) 
    for(L i=localAstroMax; i<nunkSplit; ++i){
        vVect_dev[i]=ZERO;
    }

}


void cblas_dcopy_kernel(double* wVect_dev, const double* vVect_dev, const long nunkSplit)
{

    #pragma omp target teams distribute parallel for map(to:nunkSplit) 
    for(long i=0; i<nunkSplit; ++i){
        wVect_dev[i]=vVect_dev[i];
    }

}

template<typename L, typename I>
void kAuxcopy_Kernel (double* knownTerms_dev, double* kAuxcopy_dev, const L nobs, const I N)
{

    #pragma omp target teams distribute parallel for map(to:nobs,N) 
    for(I i=0; i<N; ++i){
        kAuxcopy_dev[i]=knownTerms_dev[nobs+i];
        knownTerms_dev[nobs+i]=ZERO;

    }



}

template<typename L>
void vAuxVect_Kernel (double* vVect_dev, double* vAuxVect_dev, const L N)
{

    #pragma omp target teams distribute parallel for map(to:N) 
    for(L i=0; i<N; ++i){
        vAuxVect_dev[i]=vVect_dev[i];
        vVect_dev[i]=0;
    }


}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                                                    APROD1
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void aprod1_Kernel_astro(double*   __restrict__ knownTerms_dev, 
                        const double*   __restrict__ systemMatrix_dev, 
                        const double*   __restrict__ vVect_dev, 
                        const long*   __restrict__ matrixIndexAstro, 
                        const long offLocalAstro, 
                        const long nobs, 
                        const short nAstroPSolved){

    #ifndef AUTO_TUNING
    #pragma omp target teams distribute parallel for map(to:offLocalAstro, nobs, nAstroPSolved) num_teams(TEAMSAPROD1) thread_limit(THREADSAPROD1) 
    #else
    #pragma omp target teams distribute parallel for map(to:offLocalAstro, nobs, nAstroPSolved)  
    #endif
    for(long ix=0; ix<nobs; ++ix){
            double sum{ZERO};
            const long jstartAstro = matrixIndexAstro[ix] - offLocalAstro;
            #pragma unroll
            for(short jx = 0; jx <   nAstroPSolved; jx++) {
                sum += systemMatrix_dev[ix*nAstroPSolved + jx] * vVect_dev[jstartAstro+jx];
            }

        knownTerms_dev[ix]= knownTerms_dev[ix]+sum;

    }

}



void aprod1_Kernel_att_AttAxis(double*   __restrict__ knownTerms_dev, 
                                const double*   __restrict__ systemMatrix_dev, 
                                const double*   __restrict__ vVect_dev, 
                                const long*   __restrict__ matrixIndexAtt, 
                                const long  nAttP, 
                                const long nobs, 
                                const long nDegFreedomAtt, 
                                const int offLocalAtt, 
                                const short nAttParAxis)
{

    #ifndef AUTO_TUNING
    #pragma omp target teams distribute parallel for map(to:nAttP, offLocalAtt, nobs, nAttParAxis) num_teams(TEAMSAPROD1) thread_limit(THREADSAPROD1) 
    #else
    #pragma omp target teams distribute parallel for map(to:nAttP, offLocalAtt, nobs, nAttParAxis) 
    #endif
    for(long ix=0; ix<nobs; ++ix){

        double sum{ZERO};
        const long miValAtt{matrixIndexAtt[ix]};
        const long jstartAtt{miValAtt + offLocalAtt}; 
        #pragma unroll
        for(auto inpax = 0;inpax<nAttParAxis; ++inpax)
            sum += systemMatrix_dev[ix*nAttP + inpax ] * vVect_dev[jstartAtt + inpax];
        #pragma unroll
        for(auto inpax = 0;inpax<nAttParAxis;inpax++)
            sum+=  systemMatrix_dev[ix*nAttP+nAttParAxis+inpax] * vVect_dev[jstartAtt+nDegFreedomAtt+inpax];
        #pragma unroll
        for(auto inpax = 0;inpax<nAttParAxis;inpax++)
            sum+= systemMatrix_dev[ix*nAttP + nAttParAxis+nAttParAxis +inpax] * vVect_dev[jstartAtt+nDegFreedomAtt+nDegFreedomAtt+inpax];

        knownTerms_dev[ix]= knownTerms_dev[ix]+sum;
    }


}


void aprod1_Kernel_instr (double*   __restrict__ knownTerms_dev, 
                            const double*   __restrict__ systemMatrix_dev, 
                            const double*   __restrict__ vVect_dev, 
                            const int*   __restrict__ instrCol_dev, 
                            const long nobs, 
                            const long offLocalInstr, 
                            const short nInstrPSolved){

    #ifndef AUTO_TUNING
    #pragma omp target teams distribute parallel for map(to:nobs, offLocalInstr, nInstrPSolved) num_teams(TEAMSAPROD1) thread_limit(THREADSAPROD1) 
    #else
    #pragma omp target teams distribute parallel for map(to:nobs, offLocalInstr, nInstrPSolved) 
    #endif
    for(long ix=0; ix<nobs; ++ix){

        double sum{ZERO};
        const long iiVal{ix*nInstrPSolved};
        long ixInstr{0};
        #pragma unroll
        for(short inInstr=0;inInstr<nInstrPSolved;inInstr++){
            ixInstr=offLocalInstr+instrCol_dev[iiVal+inInstr];
            sum += systemMatrix_dev[ix * nInstrPSolved + inInstr]*vVect_dev[ixInstr];
        }

        knownTerms_dev[ix]= knownTerms_dev[ix]+sum;
    }

}


void aprod1_Kernel_glob(double*   __restrict__ knownTerms_dev, 
                        const double*   __restrict__ systemMatrix_dev, 
                        const double*   __restrict__ vVect_dev, 
                        const long offLocalGlob, 
                        const long nobs, 
                        const short nGlobP)
{
    #ifndef AUTO_TUNING
    #pragma omp target teams distribute parallel for map(to:offLocalGlob, nobs, nGlobP) num_teams(TEAMSAPROD1) thread_limit(THREADSAPROD1) 
    #else
    #pragma omp target teams distribute parallel for map(to:offLocalGlob, nobs, nGlobP) 
    #endif
    for(long ix=0; ix<nobs; ++ix){
        double sum{ZERO};
        for(short inGlob=0;inGlob<nGlobP;inGlob++){
            sum+=systemMatrix_dev[ix * nGlobP + inGlob]*vVect_dev[offLocalGlob+inGlob];
        }
        knownTerms_dev[ix]= knownTerms_dev[ix]+sum;
    }
}




/// ExtConstr
void aprod1_Kernel_ExtConstr(double*   __restrict__ knownTerms_dev, 
                            const double*   __restrict__ systemMatrix_dev, 
                            const double*   __restrict__ vVect_dev, 
                            const long VrIdAstroPDimMax, 
                            const long nobs, 
                            const long nDegFreedomAtt, 
                            const int startingAttColExtConstr, 
                            const int nEqExtConstr, 
                            const int nOfElextObs, 
                            const int numOfExtStar, 
                            const int numOfExtAttCol, 
                            const short nAstroPSolved, 
                            const short nAttAxes){

    const long offExtAttConstr{VrIdAstroPDimMax*nAstroPSolved+startingAttColExtConstr};


    #ifndef AUTO_TUNING
    #pragma omp target teams distribute parallel for map(to:nDegFreedomAtt, nobs, nEqExtConstr, nOfElextObs, numOfExtStar, numOfExtAttCol, nAstroPSolved, nAttAxes) num_teams(TEAMSAPROD1) thread_limit(THREADSAPROD1) 
    #else
    #pragma omp target teams distribute parallel for map(to:nDegFreedomAtt, nobs, nEqExtConstr, nOfElextObs, numOfExtStar, numOfExtAttCol, nAstroPSolved, nAttAxes) 
    #endif
    for(long iexc=0; iexc<nEqExtConstr; ++iexc){
            double sum{ZERO};
            const long offExtConstr{iexc*nOfElextObs};

            for(long j3=0;j3<numOfExtStar*nAstroPSolved;j3++){
                sum += systemMatrix_dev[offExtConstr+j3]*vVect_dev[j3];
            }
            for (int nax = 0; nax < nAttAxes; nax++) {
                const long offExtAtt{offExtConstr + numOfExtStar*nAstroPSolved + nax*numOfExtAttCol};
                const long vVIx{offExtAttConstr+nax*nDegFreedomAtt};

                for(long j3=0;j3<numOfExtAttCol;j3++){
                    sum += systemMatrix_dev[offExtAtt+j3]*vVect_dev[vVIx+j3];
                }
            }
            #pragma omp atomic update
            knownTerms_dev[nobs+iexc]+=sum;

    }

}

// /// BarConstr
void aprod1_Kernel_BarConstr(double*   __restrict__ knownTerms_dev, 
                            const double*   __restrict__ systemMatrix_dev, 
                            const double*   __restrict__ vVect_dev, 
                            const int nOfElextObs, 
                            const int nOfElBarObs, 
                            const int nEqExtConstr, 
                            const long nobs, 
                            const int nEqBarConstr, 
                            const int& numOfBarStar, 
                            const short nAstroPSolved){

    const long ktIx{nobs + nEqExtConstr};


    #ifndef AUTO_TUNING
    #pragma omp target teams distribute parallel for map(to:nOfElBarObs, nEqBarConstr, numOfBarStar, nAstroPSolved) num_teams(TEAMSAPROD1) thread_limit(THREADSAPROD1) 
    #else
    #pragma omp target teams distribute parallel for map(to:nOfElBarObs, nEqBarConstr, numOfBarStar, nAstroPSolved) 
    #endif
    for(int j3=0; j3<(numOfBarStar*nAstroPSolved); ++j3){
        for(int iexc=0;iexc<nEqBarConstr;iexc++ ){
            double sum{ZERO};
            const long offBarConstrIx=iexc*nOfElBarObs;
            sum = sum + systemMatrix_dev[offBarConstrIx+j3]*vVect_dev[j3];

            #pragma omp atomic update
            knownTerms_dev[ktIx+iexc]+=sum;

        }
    }
}


/// InstrConstr
void aprod1_Kernel_InstrConstr(double*   __restrict__ knownTerms_dev, 
                                const double*   __restrict__ systemMatrix_dev, 
                                const double*   __restrict__ vVect_dev, 
                                const int*   __restrict__ instrConstrIlung_dev, 
                                const int*   __restrict__ instrCol_dev, 
                                const long VrIdAstroPDimMax, 
                                const long nobs, 
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

    const long offSetInstrConstr{nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr};
    const long offSetInstrConstr1{VrIdAstroPDimMax*nAstroPSolved+nDegFreedomAtt*nAttAxes};
    const long ktIx{nobs+nEqExtConstr+nEqBarConstr};


    if(myid<nOfInstrConstr){

        #ifndef AUTO_TUNING
        #pragma omp target teams distribute parallel for map(to:nobs, myid, nOfInstrConstr, nproc, nInstrPSolved) num_teams(TEAMSAPROD1) thread_limit(THREADSAPROD1) 
        #else
        #pragma omp target teams distribute parallel for map(to:nobs, myid, nOfInstrConstr, nproc, nInstrPSolved) 
        #endif
        for(int i1=myid;i1<nOfInstrConstr;i1+=nproc){
            double sum{ZERO};
            long offSetInstrInc{offSetInstrConstr};
            int offSetInstr{0};
            for(int m=0;m<i1;m++) 
            {
                offSetInstrInc+=instrConstrIlung_dev[m];
                offSetInstr+=instrConstrIlung_dev[m];
            }
            const long offvV{nobs*nInstrPSolved+offSetInstr};
            for(int j3 = 0; j3 < instrConstrIlung_dev[i1]; j3++)
                sum+=systemMatrix_dev[offSetInstrInc+j3]*vVect_dev[offSetInstrConstr1+instrCol_dev[offvV+j3]];
            #pragma omp atomic update
            knownTerms_dev[ktIx+i1]+=sum;
        }
    }
}


//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                                                    APROD2
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void aprod2_Kernel_astro(double*   __restrict__ vVect_dev, 
                        const double*   __restrict__ systemMatrix_dev, 
                        const double*   __restrict__ knownTerms_dev, 
                        const long*   __restrict__ matrixIndexAstro, 
                        const long*   __restrict__ startend, 
                        const long offLocalAstro, 
                        const long nobs, 
                        const short nAstroPSolved){


    #ifndef AUTO_TUNING
    #pragma omp target teams distribute parallel for map(to:offLocalAstro,nobs,nAstroPSolved) num_teams(TEAMSAPROD2ASTRO) thread_limit(THREADSAPROD2ASTRO)   
    #else
    #pragma omp target teams distribute parallel for map(to:offLocalAstro,nobs,nAstroPSolved)  
    #endif
    for(long ix=0; ix<nobs; ++ix){
        long tid=matrixIndexAstro[startend[ix]];
        for(long i=startend[ix]; i<startend[ix+1]; ++i){
            #pragma unroll
            for (auto jx = 0; jx < nAstroPSolved; jx++){ 
                    vVect_dev[tid - offLocalAstro + jx]+= systemMatrix_dev[i*nAstroPSolved + jx] * knownTerms_dev[i];
            }
        }
    }
}



void aprod2_Kernel_att_AttAxis(double *  __restrict__ vVect_dev, 
                                const double *  __restrict__ systemMatrix_dev, 
                                const double *  __restrict__ knownTerms_dev, 
                                const long*   __restrict__ matrixIndexAtt, 
                                const long nAttP, 
                                const long nDegFreedomAtt, 
                                const long offLocalAtt, 
                                const long nobs, 
                                const short nAstroPSolved, 
                                const short nAttParAxis){

    #ifndef AUTO_TUNING
    #pragma omp target teams distribute parallel for map(to:offLocalAtt,nDegFreedomAtt,nAttP,nobs,nAttParAxis) num_teams(TEAMSAPROD2) thread_limit(THREADSAPROD2)  
    #else
    #pragma omp target teams distribute parallel for map(to:offLocalAtt,nDegFreedomAtt,nAttP,nobs,nAttParAxis)  
    #endif
    for(long ix=0; ix<nobs; ++ix){

        const long jstartAtt = matrixIndexAtt[ix] + offLocalAtt;
        #pragma unroll
        for (auto inpax = 0; inpax < nAttParAxis; inpax++){
            #pragma omp atomic update
            vVect_dev[jstartAtt+inpax]+=systemMatrix_dev[ix*nAttP+inpax]*knownTerms_dev[ix];
        }
        #pragma unroll
        for (auto inpax = 0; inpax < nAttParAxis; inpax++){
            #pragma omp atomic update
            vVect_dev[jstartAtt+nDegFreedomAtt+inpax]+=systemMatrix_dev[ix*nAttP+nAttParAxis+inpax]*knownTerms_dev[ix];
        }
        #pragma unroll
        for (auto inpax = 0; inpax < nAttParAxis; inpax++){
            #pragma omp atomic update
            vVect_dev[jstartAtt+nDegFreedomAtt+nDegFreedomAtt+inpax]+=systemMatrix_dev[ix*nAttP+nAttParAxis+nAttParAxis+inpax]*knownTerms_dev[ix];
        }

    }

}


void aprod2_Kernel_instr(double *  __restrict__ vVect_dev,
                        const double *  __restrict__ systemMatrix_dev, 
                        const double *  __restrict__ knownTerms_dev, 
                        const int *  __restrict__ instrCol_dev, 
                        const long offLocalInstr, 
                        const long nobs,  
                        const short nInstrPSolved){

    #ifndef AUTO_TUNING
    #pragma omp target teams distribute parallel for map(to:offLocalInstr,nobs,nInstrPSolved) num_teams(TEAMSAPROD2) thread_limit(THREADSAPROD2)   
    #else
    #pragma omp target teams distribute parallel for map(to:offLocalInstr,nobs,nInstrPSolved)   
    #endif
    for(long ix=0; ix<nobs; ++ix){
        #pragma unroll
        for (auto inInstr = 0; inInstr < nInstrPSolved; inInstr++){
            double tmp{systemMatrix_dev[ix*nInstrPSolved + inInstr] * knownTerms_dev[ix]};
            #pragma omp atomic update
            vVect_dev[offLocalInstr + instrCol_dev[ix*nInstrPSolved+inInstr]]+=tmp;
        }
    }
}



void aprod2_Kernel_ExtConstr(double*   __restrict__ vVect_dev, 
                            const double*   __restrict__ systemMatrix_dev, 
                            const double*   __restrict__ knownTerms_dev, 
                            const long nobs, 
                            const long nDegFreedomAtt, 
                            const long VrIdAstroPDimMax, 
                            const int nEqExtConstr, 
                            const int nOfElextObs, 
                            const int numOfExtStar, 
                            const int startingAttColExtConstr, 
                            const int numOfExtAttCol, 
                            const short nAstroPSolved, 
                            const short nAttAxes){

    const long off1{VrIdAstroPDimMax*nAstroPSolved+startingAttColExtConstr};

    #ifndef AUTO_TUNING
    #pragma omp target teams distribute parallel for map(to:nEqExtConstr,nobs,nOfElextObs,numOfExtStar,numOfExtAttCol,nAstroPSolved) num_teams(TEAMSAPROD2) thread_limit(THREADSAPROD2) 
    #else
    #pragma omp target teams distribute parallel for map(to:nEqExtConstr,nobs,nOfElextObs,numOfExtStar,numOfExtAttCol,nAstroPSolved) 
    #endif
    for(int i=0; i<numOfExtStar; ++i){
        const long off3{i*nAstroPSolved};        
        for(int ix = 0; ix < nEqExtConstr; ++ix){  
            const double yi{knownTerms_dev[nobs + ix]};
            const long offExtStarConstrEq{ix*nOfElextObs};
            const long off2{offExtStarConstrEq + off3};
                for(int j2 = 0; j2 < nAstroPSolved; ++j2){
                    vVect_dev[j2+off3] += systemMatrix_dev[off2+j2]*yi;
                }
            
        } 
    }

    #ifndef AUTO_TUNING
    #pragma omp target teams distribute parallel for map(to:nDegFreedomAtt,nEqExtConstr,nobs,nOfElextObs,numOfExtStar,numOfExtAttCol,nAstroPSolved,nAttAxes) num_teams(TEAMSAPROD2) thread_limit(THREADSAPROD2) 
    #else
    #pragma omp target teams distribute parallel for map(to:nDegFreedomAtt,nEqExtConstr,nobs,nOfElextObs,numOfExtStar,numOfExtAttCol,nAstroPSolved,nAttAxes) 
    #endif
    for(int i=0; i<numOfExtAttCol; ++i){
        for(int ix=0;ix<nEqExtConstr;ix++ ){  
            const double yi = knownTerms_dev[nobs + ix];
            const long offExtAttConstrEq{ix*nOfElextObs + numOfExtStar*nAstroPSolved}; 
            for(int nax = 0; nax < nAttAxes; nax++){
                const long off2{offExtAttConstrEq+nax*numOfExtAttCol};
                const long offExtUnk{off1 + nax*nDegFreedomAtt};
                vVect_dev[offExtUnk+i] = vVect_dev[offExtUnk+i] + systemMatrix_dev[off2+i]*yi;
            }
        }
    }
}


void aprod2_Kernel_BarConstr(double*   __restrict__ vVect_dev, 
                            const double*   __restrict__ systemMatrix_dev, 
                            const double*   __restrict__ knownTerms_dev, 
                            const long nobs, 
                            const int nEqBarConstr, 
                            const int nEqExtConstr, 
                            const int nOfElextObs, 
                            const int nOfElBarObs, 
                            const int numOfBarStar, 
                            const short nAstroPSolved){
    

    #ifndef AUTO_TUNING
    #pragma omp target teams distribute parallel for map(to: nEqBarConstr, nobs, nEqExtConstr, nOfElextObs, nOfElBarObs, numOfBarStar, nAstroPSolved) num_teams(TEAMSAPROD2) thread_limit(THREADSAPROD2) 
    #else
    #pragma omp target teams distribute parallel for map(to: nEqBarConstr, nobs, nEqExtConstr, nOfElextObs, nOfElBarObs, numOfBarStar, nAstroPSolved) 
    #endif
    for(int yx=0; yx<numOfBarStar; ++yx){
        for(int ix=0;ix<nEqBarConstr;++ix){  
            const double yi{knownTerms_dev[nobs+nEqExtConstr+ix]};
            const long offBarStarConstrEq{nEqExtConstr*nOfElextObs+ix*nOfElBarObs};
            for(auto j2=0;j2<nAstroPSolved;j2++){
                vVect_dev[yx*nAstroPSolved + j2] += systemMatrix_dev[offBarStarConstrEq+yx*nAstroPSolved+j2]*yi;
            }
        }
    }


}

void aprod2_Kernel_InstrConstr(double*   __restrict__ vVect, 
                                const double*   __restrict__ systemMatrix, 
                                const double*   __restrict__ knownTerms, 
                                const int*   __restrict__ instrConstrIlung, 
                                const int*   __restrict__ instrCol, 
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
                                const short nInstrPSolved, 
                                const short nAstroPSolved, 
                                const short nAttAxes){


    if(myid<nOfInstrConstr){
        const long off3{nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr};
        const long offInstrUnk{VrIdAstroPDimMax*nAstroPSolved+nDegFreedomAtt*nAttAxes};
        const long off2{mapNoss+nEqExtConstr+nEqBarConstr};
        const long off4{mapNoss*nInstrPSolved};


        #ifndef AUTO_TUNING
        #pragma omp target teams distribute parallel for map(to: myid, nOfInstrConstr, nproc) num_teams(TEAMSAPROD2) thread_limit(THREADSAPROD2) is_device_ptr(knownTerms)
        #else
        #pragma omp target teams distribute parallel for map(to: myid, nOfInstrConstr, nproc) is_device_ptr(knownTerms)
        #endif
        for(int k1_Aux=myid;k1_Aux<nOfInstrConstr;k1_Aux+=nproc){
            const double yi{knownTerms[off2+k1_Aux]};
            int offSetInstr=0;
            for(int m=0;m<k1_Aux;++m){
                offSetInstr=offSetInstr+instrConstrIlung[m];
            }
            const long off1{off3+offSetInstr};
            const long off5{off4+offSetInstr};
            for(int j=0;j<instrConstrIlung[k1_Aux];j++){
                #pragma omp atomic update
                vVect[offInstrUnk+instrCol[off5+j]]+=systemMatrix[off1+j]*yi;
            }
        }
    }


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


    const int myid{comlsqr.myid};
    const int num_gpus = omp_get_num_devices();
    if(num_gpus>1)  my_gpu =myid%num_gpus;
    const int init_dev=omp_get_initial_device();

    //-----------------------------------------------------------------------
    ///////////// Specific definitions
    double startCycleTime,endCycleTime,totTime;
    int nAstroElements;

    ////////////////////////////////	
    //  Initialize.
    const long mapNoss{static_cast<long>(comlsqr.mapNoss[myid])};
    const long VrIdAstroPDimMax=comlsqr.VrIdAstroPDimMax; 
    const long offsetAttParam = comlsqr.offsetAttParam;
    const long offsetInstrParam = comlsqr.offsetInstrParam;
    const long offsetGlobParam = comlsqr.offsetGlobParam;  
    const long localAstroMax = VrIdAstroPDimMax * comlsqr.nAstroPSolved; 
    const long offLocalAstro = comlsqr.mapStar[myid][0] * comlsqr.nAstroPSolved;
    const long offLocalInstr = offsetInstrParam + (localAstroMax - offsetAttParam); 
    const long offLocalGlob = offsetGlobParam + (localAstroMax - offsetAttParam); 
    const long nunkSplit=comlsqr.nunkSplit;
    const long VrIdAstroPDim=comlsqr.VrIdAstroPDim;  
    const long nDegFreedomAtt=comlsqr.nDegFreedomAtt;
    const long localAstro=VrIdAstroPDim*comlsqr.nAstroPSolved;
    const long int offLocalAtt = localAstroMax - offsetAttParam; 

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

    const int other{static_cast<int>(nAttParam + nInstrParam + nGlobalParam)}; 


    long nElemKnownTerms = mapNoss+nEqExtConstr+nEqBarConstr+nOfInstrConstr;
    int nTotConstraints  =   nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr+nElemIC;


    double max_knownTerms{ZERO};
    double ssq_knownTerms{ZERO};
    double max_vVect{ZERO};
    double ssq_vVect{ZERO};

    double alpha{ZERO};
    double alphaLoc{ZERO};
    double alphaLoc2{ZERO};
    double temp{ZERO};


    double beta{ZERO};
    double betaLoc{ZERO};
    double betaLoc2{ZERO};

    double anorm{ZERO};
    double acond{ZERO};
    double rnorm{ZERO};
    double arnorm{ZERO};
    double xnorm{ZERO};
    double rhobar{ZERO};
    double phibar{ZERO};
    double bnorm{ZERO};

    double rhbar1,cs1,sn1,psi,rho,cs,sn,theta,phi,tau,t1,t2,t3,dknorm;
    double dnorm, dxk, dxmax, delta, gambar, rhs, zbar, gamma, cs2, sn2,z,xnorm1;
    double res2, test1,test3,rtol,ctol;
    double test2{ZERO};

    int itn=0,istop=0;

    const bool damped = damp > ZERO;
    const bool wantse = standardError != NULL;

    double t{ZERO};


    #ifdef VERBOSE
        const int noCov{comlsqr.noCov};
        double alfopt;
        int  maxdx=0;
    #endif




    double* kAuxcopy = new double[nEqExtConstr+nEqBarConstr+nOfInstrConstr]();
    double* vAuxVect = new double[localAstroMax](); 

    //--------------------------------------------------------------------------------------------------------
    long nnz=1;
    for(long i=0; i<mapNoss-1; i++){
        if(matrixIndexAstro[i]!=matrixIndexAstro[i+1]){
            nnz++;
        }
    }


    long *startend=(long*)malloc(sizeof(long)*(nnz+1));

    long count=0; nnz=0;
    startend[nnz]=count;nnz++;
    for(long i=0; i<mapNoss-1; ++i){
        if(matrixIndexAstro[i]!=matrixIndexAstro[i+1]){
            count++;
            startend[nnz]=count;
            nnz++;
        }else{
            count++;
        }
    }
    startend[nnz]=count+1;
    //--------------------------------------------------------------------------------------------------------


    #pragma omp target enter data map(to:                                                                              \
                                        sysmatAstro[0:mapNoss*nAstroPSolved],                                          \
                                        sysmatAtt[0:mapNoss*nAttP],                                                    \
                                        sysmatInstr[0:mapNoss*nInstrPSolved],                                          \
                                        sysmatGloB[0:mapNoss*nGlobP],                                                  \
                                        sysmatConstr[0:nTotConstraints],                                               \
                                        vVect[0:nunkSplit],                                                            \
                                        wVect[0:nunkSplit],                                                            \
                                        matrixIndexAstro[0:mapNoss],                                                   \
                                        matrixIndexAtt[0:mapNoss],                                                     \
                                        startend[0:nnz+1],                                                             \
                                        instrCol[0:nInstrPSolved*mapNoss+nElemIC],                                     \
                                        instrConstrIlung[0:nOfInstrConstr],                                            \
                                        kAuxcopy[0:nEqExtConstr+nEqBarConstr+nOfInstrConstr],                          \
                                        vAuxVect[0:localAstroMax],                                                     \
                                        xSolution[0:nunkSplit],                                                        \
                                        standardError[0:nunkSplit]                                                     \
					) 


    double* knownTerms_dev=(double*)omp_target_alloc(nElemKnownTerms*sizeof(double), my_gpu);
    omp_target_memcpy(knownTerms_dev,knownTerms, nElemKnownTerms*sizeof (double),0,0,my_gpu,init_dev);

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


    dload(vVect, nunkSplit, ZERO);

    #pragma omp target update to(vVect[0:nunkSplit]) 

    dload(xSolution,nunkSplit, ZERO);

    if ( wantse )   dload(standardError,nunkSplit, ZERO );

    #pragma omp target update to(xSolution[0:nunkSplit]) 
    #pragma omp target update to(standardError[0:nunkSplit]) 


    max_knownTerms = maxCommMultiBlock_double(knownTerms_dev,nElemKnownTerms);

    if(!myid){
        ssq_knownTerms = sumCommMultiBlock_double(knownTerms_dev, mapNoss + nEqExtConstr + nEqBarConstr+nOfInstrConstr, max_knownTerms);
        betaLoc = max_knownTerms*std::sqrt(ssq_knownTerms);
    }else{
        ssq_knownTerms = sumCommMultiBlock_double(knownTerms_dev, mapNoss, max_knownTerms);
        betaLoc = max_knownTerms*std::sqrt(ssq_knownTerms);
    }

    betaLoc2=betaLoc*betaLoc;

    //------------------------------------------------------------------------------------------------  TIME 2
    starttime=MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE,&betaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    communicationtime+=MPI_Wtime()-starttime;
    //------------------------------------------------------------------------------------------------
    beta=std::sqrt(betaLoc2);

    if(beta > ZERO){


        dscal(knownTerms_dev,1.0/beta,nElemKnownTerms,ONE);

        if(myid) vVect_Put_To_Zero_Kernel(vVect,localAstroMax,nunkSplit);

        //APROD2 CALL BEFORE LSQR
        if(nAstroPSolved) aprod2_Kernel_astro(vVect, sysmatAstro, knownTerms_dev, matrixIndexAstro, startend, offLocalAstro, nnz, nAstroPSolved);
        if(nAttP) aprod2_Kernel_att_AttAxis(vVect,sysmatAtt,knownTerms_dev,matrixIndexAtt,nAttP,nDegFreedomAtt,offLocalAtt,mapNoss,nAstroPSolved,nAttParAxis);
        if(nInstrPSolved) aprod2_Kernel_instr(vVect,sysmatInstr, knownTerms_dev, instrCol, offLocalInstr, mapNoss, nInstrPSolved);
        // // NOT OPTIMIZED YET
        if(nEqExtConstr) aprod2_Kernel_ExtConstr(vVect,sysmatConstr,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAstroPSolved,nAttAxes);
        if(nEqBarConstr) aprod2_Kernel_BarConstr(vVect,sysmatConstr,knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
        if(nOfInstrConstr) aprod2_Kernel_InstrConstr(vVect, sysmatConstr, knownTerms_dev, instrConstrIlung, instrCol,  VrIdAstroPDimMax, nDegFreedomAtt, mapNoss, nEqExtConstr, nEqBarConstr, nOfElextObs, nOfElBarObs, myid, nOfInstrConstr, nproc, nInstrPSolved, nAstroPSolved, nAttAxes);


        #pragma omp target update from(vVect[0:nunkSplit]) 

        //------------------------------------------------------------------------------------------------  TIME4
        starttime=MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE,&vVect[localAstroMax],static_cast<long>(nAttParam+nInstrParam+nGlobalParam),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        //------------------------------------------------------------------------------------------------
        if(nAstroPSolved) SumCirc2(vVect,comlsqr,&communicationtime);
        //------------------------------------------------------------------------------------------------

        #pragma omp target update to(vVect[0:nunkSplit]) 


        nAstroElements=comlsqr.mapStar[myid][1]-comlsqr.mapStar[myid][0] + 1;
        if(myid<nproc-1)
        {
            nAstroElements=comlsqr.mapStar[myid][1]-comlsqr.mapStar[myid][0] +1;
            if(comlsqr.mapStar[myid][1]==comlsqr.mapStar[myid+1][0]) nAstroElements--;
        }


        max_vVect = maxCommMultiBlock_double(vVect, nunkSplit);
        ssq_vVect = sumCommMultiBlock_double(vVect, nAstroElements*nAstroPSolved, max_vVect);

        alphaLoc = max_vVect*std::sqrt(ssq_vVect);
        alphaLoc2=alphaLoc*alphaLoc;


        if(!myid){
            double alphaOther2 = alphaLoc2;
            ssq_vVect = sumCommMultiBlock_double(&vVect[localAstroMax], nunkSplit - localAstroMax, max_vVect);
            alphaLoc = max_vVect*std::sqrt(ssq_vVect);
            alphaLoc2 = alphaLoc*alphaLoc;
            alphaLoc2 = alphaOther2 + alphaLoc2;
        }

        //------------------------------------------------------------------------------------------------  TIME6
        starttime=MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE,&alphaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //------------------------------------------------------------------------------------------------
        alpha=std::sqrt(alphaLoc2);
    }


    if(alpha > ZERO){
        dscal(vVect, 1/alpha, nunkSplit, ONE);
        cblas_dcopy_kernel(wVect,vVect,nunkSplit);
    }

    #pragma omp target update from(vVect[0:nunkSplit],wVect[0:nunkSplit]) 
    
    omp_target_memcpy (knownTerms, knownTerms_dev, nElemKnownTerms*sizeof (double),0,0,init_dev, my_gpu);

    arnorm  = alpha * beta;

    if (arnorm == ZERO){

        #pragma omp target exit data map(delete:        \
                                    sysmatAstro,        \
                                    sysmatAtt,          \
                                    sysmatInstr,        \
                                    sysmatGloB,         \
                                    sysmatConstr,       \
                                    vVect,              \
                                    wVect,              \
                                    matrixIndexAstro,   \
                                    matrixIndexAtt,     \
                                    startend,           \
                                    instrCol,           \
                                    instrConstrIlung,   \
                                    kAuxcopy,           \
                                    vAuxVect) 


        *istop_out  = istop;
        *itn_out    = itn;
        *anorm_out  = anorm;
        *acond_out  = acond;
        *rnorm_out  = rnorm;
        *arnorm_out = test2;
        *xnorm_out  = xnorm;

        delete [] kAuxcopy;
        delete [] vAuxVect;

        return ;

    } 
    rhobar =   alpha;
    phibar =   beta;
    bnorm  =   beta;
    rnorm  =   beta;

    if(!myid){
        test1  = ONE;
        test2  = alpha / beta;
    }



    #pragma omp target update to(vVect[0:nunkSplit],wVect[0:nunkSplit]) 
    omp_target_memcpy (knownTerms_dev,knownTerms, nElemKnownTerms*sizeof (double),0,0,my_gpu,init_dev);


    TimeSetInit=MPI_Wtime()-TimeSetInit;
    LoopTime=startCycleTime=MPI_Wtime(); 



    //  ==================================================================
    //  ==================================================================
    //                          MAIN ITERATION LOOP
    //  ==================================================================
    //  ==================================================================
    if (!myid) std::cout<<"LSQR: START ITERATIONS"<<std::endl; 
        
    startCycleTime=MPI_Wtime();


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
        

        ++itn;
        comlsqr.itn=itn;

        dscal(knownTerms_dev, alpha, nElemKnownTerms, MONE);
        kAuxcopy_Kernel(knownTerms_dev, kAuxcopy, mapNoss, nEqExtConstr+nEqBarConstr+nOfInstrConstr);

        // //////////////////////////////////// APROD MODE 1
        // //APROD1 ASTRO CALL
        if(nAstroPSolved) aprod1_Kernel_astro(knownTerms_dev, sysmatAstro, vVect, matrixIndexAstro, offLocalAstro, mapNoss, nAstroPSolved);
        if(nAttP) aprod1_Kernel_att_AttAxis(knownTerms_dev,sysmatAtt,vVect,matrixIndexAtt,nAttP,mapNoss,nDegFreedomAtt,offLocalAtt,nAttParAxis);
        if(nInstrPSolved) aprod1_Kernel_instr(knownTerms_dev, sysmatInstr, vVect, instrCol, mapNoss, offLocalInstr, nInstrPSolved);
        if(nGlobP) aprod1_Kernel_glob(knownTerms_dev, sysmatGloB, vVect, offLocalGlob, mapNoss, nGlobP);
        // //////////////////////////////////// CONSTRAINTS APROD MODE 1        
        if(nEqExtConstr) aprod1_Kernel_ExtConstr(knownTerms_dev,sysmatConstr,vVect,VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,startingAttColExtConstr,nEqExtConstr,nOfElextObs,numOfExtStar,numOfExtAttCol,nAstroPSolved,nAttAxes);
        if(nEqBarConstr) aprod1_Kernel_BarConstr(knownTerms_dev, sysmatConstr, vVect, nOfElextObs, nOfElBarObs, nEqExtConstr, mapNoss, nEqBarConstr, numOfBarStar, nAstroPSolved );
        if(nOfInstrConstr) aprod1_Kernel_InstrConstr(knownTerms_dev,sysmatConstr,vVect,instrConstrIlung,instrCol,VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,nOfElextObs,nEqExtConstr,nOfElBarObs,nEqBarConstr,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);

        omp_target_memcpy (&knownTerms[mapNoss],&knownTerms_dev[mapNoss],(nEqExtConstr+nEqBarConstr+nOfInstrConstr)*sizeof(double),0,0,init_dev, my_gpu);

        //------------------------------------------------------------------------------------------------
        starttime=MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE,&knownTerms[mapNoss],nEqExtConstr+nEqBarConstr+nOfInstrConstr,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //------------------------------------------------------------------------------------------------
        omp_target_memcpy (&knownTerms_dev[mapNoss],&knownTerms[mapNoss],(nEqExtConstr+nEqBarConstr+nOfInstrConstr)*sizeof(double),0,0,my_gpu,init_dev);


        #pragma omp target teams distribute parallel for 
        for(int i=0;i<nEqExtConstr+nEqBarConstr+nOfInstrConstr;++i)
        {
            knownTerms_dev[mapNoss+i] += kAuxcopy[i];
        }

        max_knownTerms = maxCommMultiBlock_double(knownTerms_dev,nElemKnownTerms);

        if(!myid){
            ssq_knownTerms = sumCommMultiBlock_double(knownTerms_dev, mapNoss + nEqExtConstr + nEqBarConstr+nOfInstrConstr, max_knownTerms);
            betaLoc = max_knownTerms*std::sqrt(ssq_knownTerms);
            betaLoc2 = betaLoc*betaLoc;
        }else{
            ssq_knownTerms = sumCommMultiBlock_double(knownTerms_dev, mapNoss, max_knownTerms);
            betaLoc = max_knownTerms*std::sqrt(ssq_knownTerms);
            betaLoc2 = betaLoc*betaLoc;

        }

        //------------------------------------------------------------------------------------------------
        starttime=MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE,&betaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //------------------------------------------------------------------------------------------------
        beta=std::sqrt(betaLoc2);

        temp   =   d2norm( alpha, beta );
        temp   =   d2norm( temp , damp );
        anorm  =   d2norm( anorm, temp );

        if(beta>ZERO){

            dscal(knownTerms_dev,1/beta, nElemKnownTerms,ONE);
            dscal(vVect,beta,nunkSplit,MONE);
            vAuxVect_Kernel(vVect, vAuxVect, localAstroMax);


            if (myid) {
                vVect_Put_To_Zero_Kernel(vVect,localAstroMax,nunkSplit);
            }


            //APROD2 CALL BEFORE LSQR
            if(nAstroPSolved) aprod2_Kernel_astro(vVect, sysmatAstro, knownTerms_dev, matrixIndexAstro, startend, offLocalAstro, nnz, nAstroPSolved);
            if(nAttP) aprod2_Kernel_att_AttAxis(vVect,sysmatAtt,knownTerms_dev,matrixIndexAtt,nAttP,nDegFreedomAtt,offLocalAtt,mapNoss,nAstroPSolved,nAttParAxis);
            if(nInstrPSolved) aprod2_Kernel_instr(vVect,sysmatInstr, knownTerms_dev, instrCol, offLocalInstr, mapNoss, nInstrPSolved);
            // // NOT OPTIMIZED YET
            if(nEqExtConstr) aprod2_Kernel_ExtConstr(vVect,sysmatConstr,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAstroPSolved,nAttAxes);
            if(nEqBarConstr) aprod2_Kernel_BarConstr(vVect,sysmatConstr,knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
            if(nOfInstrConstr) aprod2_Kernel_InstrConstr(vVect, sysmatConstr, knownTerms_dev, instrConstrIlung, instrCol,  VrIdAstroPDimMax, nDegFreedomAtt, mapNoss, nEqExtConstr, nEqBarConstr, nOfElextObs, nOfElBarObs, myid, nOfInstrConstr, nproc, nInstrPSolved, nAstroPSolved, nAttAxes);

            #pragma omp target update from(vVect[0:nunkSplit]) 



            //------------------------------------------------------------------------------------------------
            starttime=MPI_Wtime();
                MPI_Allreduce(MPI_IN_PLACE,&vVect[localAstroMax],static_cast<long>(nAttParam+nInstrParam+nGlobalParam),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  
            communicationtime+=MPI_Wtime()-starttime;
            //------------------------------------------------------------------------------------------------


            //------------------------------------------------------------------------------------------------
            if(nAstroPSolved) SumCirc2(vVect,comlsqr,&communicationtime);
        	// if(nAstroPSolved) SumCirc(vVect,comlsqr);
            //------------------------------------------------------------------------------------------------

            #pragma omp target update to(vVect[0:nunkSplit]) 

            
            #pragma omp target teams distribute parallel for 
            for(long i=0; i < localAstroMax; i++) {
                vVect[i] += vAuxVect[i];
            }
                                    
            max_vVect = maxCommMultiBlock_double(vVect, nunkSplit);
            ssq_vVect = sumCommMultiBlock_double(vVect, nAstroElements*nAstroPSolved, max_vVect);

            double alphaLoc{ZERO};
            alphaLoc = max_vVect*std::sqrt(ssq_vVect);
            alphaLoc2=alphaLoc*alphaLoc;



            if(!myid){
                double alphaOther2 = alphaLoc2;
                ssq_vVect = sumCommMultiBlock_double(&vVect[localAstroMax], nunkSplit - localAstroMax, max_vVect);
                alphaLoc = max_vVect*std::sqrt(ssq_vVect);
                alphaLoc2 = alphaLoc*alphaLoc;
                alphaLoc2 = alphaOther2 + alphaLoc2;
            }


            //------------------------------------------------------------------------------------------------
            starttime=MPI_Wtime();
                MPI_Allreduce(MPI_IN_PLACE,&alphaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            communicationtime+=MPI_Wtime()-starttime;
            //------------------------------------------------------------------------------------------------

            alpha=std::sqrt(alphaLoc2);
                    

            if (alpha > ZERO) {
                dscal(vVect,1/alpha,nunkSplit,ONE);
            }

        }

        //------------------------------------------------------------------
        //      Use a plane rotation to eliminate the damping parameter.
        //      This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        //------------------------------------------------------------------
        rhbar1 = rhobar;
        if ( damped ) {
            rhbar1 = d2norm( rhobar, damp );
            cs1    = rhobar / rhbar1;
            sn1    = damp   / rhbar1;
            psi    = sn1 * phibar;
            phibar = cs1 * phibar;
        }

        //------------------------------------------------------------------
        //      Use a plane rotation to eliminate the subdiagonal element (beta)
        //      of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        //------------------------------------------------------------------
        rho    =   d2norm( rhbar1, beta );
        cs     =   rhbar1 / rho;
        sn     =   beta   / rho;
        theta  =   sn * alpha;
        rhobar = - cs * alpha;
        phi    =   cs * phibar;
        phibar =   sn * phibar;
        tau    =   sn * phi;


        //------------------------------------------------------------------
        //      Update  x, w  and (perhaps) the standard error estimates.
        //------------------------------------------------------------------
        t1     =   phi   / rho;
        t2     = - theta / rho;
        t3     =   ONE   / rho;
        dknorm =   ZERO;

        #pragma omp target teams distribute parallel for map(to:t3,nAstroElements,nAstroPSolved) map(tofrom:dknorm) num_teams(blockSize) num_threads(gridSize) reduction(+:dknorm)
        for (long i = 0; i < nAstroElements*nAstroPSolved; i++) {
            t      =  wVect[i];
            t      = (t3*t)*(t3*t);
            dknorm =  t     +  dknorm;
        }

        //------------------------------------------------------------------------------------------------
        starttime=MPI_Wtime();
         		MPI_Allreduce(MPI_IN_PLACE,&dknorm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //------------------------------------------------------------------------------------------------ 		

        // #pragma omp target update from(vVect[0:nunkSplit]) 

        #pragma omp target teams distribute parallel for map(to:localAstro,t1) num_teams(blockSize) num_threads(gridSize) 
        for (long i = 0; i < localAstro; i++) 
            xSolution[i]   =  t1*wVect[i]  +  xSolution[i];

        if(wantse){
            #pragma omp target teams distribute parallel for map(to:localAstro,t3) num_teams(blockSize) num_threads(gridSize) 
            for (long i = 0; i < localAstro; i++)
                standardError[i]  =  standardError[i]+(t3*wVect[i])*(t3*wVect[i]);
        }

        #pragma omp target teams distribute parallel for map(to:localAstro,t2) num_teams(blockSize) num_threads(gridSize)
        for (long i = 0; i < localAstro; i++) 
            wVect[i]   =  t2*wVect[i]  +  vVect[i];

        #pragma omp target teams distribute parallel for map(to:localAstroMax,other,t1) num_teams(blockSize) num_threads(gridSize)
        for (long i = localAstroMax; i < localAstroMax+other; i++) 
            xSolution[i]   =  xSolution[i] + t1*wVect[i]; 

        if(wantse){
            #pragma omp target teams distribute parallel for map(to:localAstroMax,other,t3) num_teams(blockSize) num_threads(gridSize)
            for (long i = localAstroMax; i < localAstroMax+other; i++)
                standardError[i]  =  standardError[i]+(t3*wVect[i])*(t3*wVect[i]);
        }

        #pragma omp target teams distribute parallel for map(to:localAstroMax,other,t2) num_teams(blockSize) num_threads(gridSize)
        for (long i = localAstroMax; i < localAstroMax+other; i++) 
            wVect[i]   =  t2*wVect[i]  +  vVect[i];
        
        #pragma omp target teams distribute parallel for map(to:localAstroMax,other,t3) map(tofrom:dknorm) num_teams(blockSize) num_threads(gridSize) reduction(+:dknorm)
        for (long i = localAstroMax; i < localAstroMax+other; i++) {
            t      =  wVect[i];
            t      = (t3*t)*(t3*t);
            dknorm =  t     +  dknorm;
        }


        //------------------------------------------------------------------
        //      Monitor the norm of d_k, the update to x.
        //      dknorm = norm( d_k )
        //      dnorm  = norm( D_k ),        where   D_k = (d_1, d_2, ..., d_k )
        //      dxk    = norm( phi_k d_k ),  where new x = x_k + phi_k d_k.
        //------------------------------------------------------------------
        dknorm = std::sqrt( dknorm );
        dnorm  = d2norm( dnorm, dknorm );
        dxk    = std::fabs( phi * dknorm );
        if (dxmax < dxk ) {
            dxmax   =  dxk;
            #ifdef VERBOSE
                maxdx   =  itn;
            #endif
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
        arnorm =   alpha * std::fabs( tau );

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

        //      Allow for tolerances set by the user.
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

    #pragma omp target update from(xSolution[0:nunkSplit],standardError[0:nunkSplit]) 


    //------------------------------------------------------------------------------------------------ 		
    if ( wantse ) {
        t    =   ONE;
        if (m > n)     t = m - n;
        if ( damped )  t = m;
        t    =   rnorm / sqrt( t );
      
        for (long i = 0; i < nunkSplit; i++)
            standardError[i]  = t * std::sqrt( standardError[i] );
        
    }



    #pragma omp target exit data map(delete:        \
                                sysmatAstro,        \
                                sysmatAtt,          \
                                sysmatInstr,        \
                                sysmatGloB,         \
                                sysmatConstr,       \
                                vVect,              \
                                wVect,              \
                                matrixIndexAstro,   \
                                matrixIndexAtt,     \
                                startend,           \
                                instrCol,           \
                                instrConstrIlung,   \
                                kAuxcopy,           \
                                vAuxVect,           \
                                xSolution,          \
                                standardError) 


    omp_target_free(knownTerms_dev, my_gpu);

    *istop_out  = istop;
    *itn_out    = itn;
    *anorm_out  = anorm;
    *acond_out  = acond;
    *rnorm_out  = rnorm;
    *arnorm_out = test2;
    *xnorm_out  = xnorm;

    delete [] kAuxcopy;
    delete [] vAuxVect;

    return;

}


