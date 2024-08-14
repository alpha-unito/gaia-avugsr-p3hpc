#include <stdlib.h>
#include <stdio.h>
#include "util.h"
#include <mpi.h>
#include <math.h>

#define __MSGSIZ_MAX 100000




int err_malloc(const char *s,int id) {
	printf("out of memory while allocating %s on PE=%d.\n", s, id);
	MPI_Abort(MPI_COMM_WORLD, 1);
	return 1;
}



void SumCirc2(double *vectToSum,struct comData comlsqr, double* communicationtime)
{
    double starttime;

	int rank, size,  npeSend, npeRecv;
	MPI_Status status; 
	MPI_Request req2,req3;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    int multMI=2;
	int nMov=2;
	if(size==2) nMov=1;
	if(size==1) return;

	double *tempSendBuf, *tempRecvBuf;
	int tempSendIdBuf[2],tempRecvIdBuf[2];


	tempSendBuf=(double *) calloc(multMI*comlsqr.nAstroPSolved,sizeof(double));
	tempRecvBuf=(double *) calloc(multMI*comlsqr.nAstroPSolved,sizeof(double));

	npeSend=rank+1;
	if(npeSend==size) npeSend=0;
	npeRecv=rank-1;
	if(npeRecv<0) npeRecv=size-1;

	tempSendIdBuf[0]=comlsqr.mapStar[rank][0]; //strating star
	tempSendIdBuf[1]=comlsqr.mapStar[rank][1]; //ending star

	
	for(int i=0;i<comlsqr.nAstroPSolved;i++)
		tempSendBuf[i]=vectToSum[i];
	for(int i=0;i<comlsqr.nAstroPSolved;i++)
		tempSendBuf[i+comlsqr.nAstroPSolved]=vectToSum[(comlsqr.VrIdAstroPDim-1)*comlsqr.nAstroPSolved+i];


    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    //------------------------------------------------------------------------------------------------

	for(int i=0;i<nMov;i++)
	{
		if(i==0) //forward propagation!
		{
			npeSend=rank+1;
			if(npeSend==size) npeSend=0;
			npeRecv=rank-1;
			if(npeRecv<0) npeRecv=size-1;
		}
		if(i==1) //backward propagation!
		{
			npeSend=rank-1;
			if(npeSend<0) npeSend=size-1;
			npeRecv=rank+1;
			if(npeRecv==size) npeRecv=0;
		}

        //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        //------------------------------------------------------------------------------------------------
        starttime=MPI_Wtime();

		MPI_Isend(tempSendIdBuf, multMI, MPI_INT, npeSend, 1,MPI_COMM_WORLD, &req2);
		MPI_Isend(tempSendBuf, multMI*comlsqr.nAstroPSolved, MPI_DOUBLE, npeSend, 2,MPI_COMM_WORLD, &req3);

		MPI_Recv(tempRecvIdBuf, multMI, MPI_INT, npeRecv, 1,MPI_COMM_WORLD, &status);
		MPI_Recv(tempRecvBuf, multMI*comlsqr.nAstroPSolved, MPI_DOUBLE, npeRecv, 2,MPI_COMM_WORLD, &status);

		MPI_Wait(&req2,&status);
		MPI_Wait(&req3,&status);


        *communicationtime+=MPI_Wtime()-starttime;
        //------------------------------------------------------------------------------------------------

		
		int okupd=0;
		if(tempRecvIdBuf[1]==comlsqr.mapStar[rank][0])
		{
		   for(int ns=0;ns<comlsqr.nAstroPSolved;ns++)
					vectToSum[ns]+=tempRecvBuf[comlsqr.nAstroPSolved+ns];
		   okupd=1;
		}
		if(tempRecvIdBuf[1]==comlsqr.mapStar[rank][1] && okupd==0)
		{
		   for(int ns=0;ns<comlsqr.nAstroPSolved;ns++)
					vectToSum[(comlsqr.VrIdAstroPDim-1)*comlsqr.nAstroPSolved+ns]+=tempRecvBuf[comlsqr.nAstroPSolved+ns];
		}
		
	    okupd=0;
		
		if(tempRecvIdBuf[0]!=tempRecvIdBuf[1])
		{
		
		  if(tempRecvIdBuf[0]==comlsqr.mapStar[rank][1] )
 		  {
			for(int ns=0;ns<comlsqr.nAstroPSolved;ns++)
				vectToSum[(comlsqr.VrIdAstroPDim-1)*comlsqr.nAstroPSolved+ns]+=tempRecvBuf[ns];
			okupd=1;
           }
            if(tempRecvIdBuf[0]==comlsqr.mapStar[rank][0])
            {
            for(int ns=0;ns<comlsqr.nAstroPSolved;ns++)
                        vectToSum[ns]+=tempRecvBuf[ns];
            }
		
		}
		
	} 
    	
		

    free(tempSendBuf); 
    free(tempRecvBuf);


}


void SumCirc(double *vectToSum,struct comData comlsqr)
{
	int rank, size,  npeSend, npeRecv;
	MPI_Status status; 
	MPI_Request req2,req3;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int nMov=2;
	if(size==2) nMov=1;
	if(size==1) return;

	double *tempSendBuf, *tempRecvBuf;
	int tempSendIdBuf[2],tempRecvIdBuf[2];

    int multMI=2;

	tempSendBuf=(double *) calloc(multMI*comlsqr.nAstroPSolved,sizeof(double));
	tempRecvBuf=(double *) calloc(multMI*comlsqr.nAstroPSolved,sizeof(double));

	npeSend=rank+1;
	if(npeSend==size) npeSend=0;
	npeRecv=rank-1;
	if(npeRecv<0) npeRecv=size-1;

	tempSendIdBuf[0]=comlsqr.mapStar[rank][0]; //strating star
	tempSendIdBuf[1]=comlsqr.mapStar[rank][1]; //ending star

	
	for(int i=0;i<comlsqr.nAstroPSolved;i++)
		tempSendBuf[i]=vectToSum[i];
	for(int i=0;i<comlsqr.nAstroPSolved;i++)
		tempSendBuf[i+comlsqr.nAstroPSolved]=vectToSum[(comlsqr.VrIdAstroPDim-1)*comlsqr.nAstroPSolved+i];

	for(int i=0;i<nMov;i++)
	{
		if(i==0) //forward propagation!
		{
			npeSend=rank+1;
			if(npeSend==size) npeSend=0;
			npeRecv=rank-1;
			if(npeRecv<0) npeRecv=size-1;
		}
		if(i==1) //backward propagation!
		{
			npeSend=rank-1;
			if(npeSend<0) npeSend=size-1;
			npeRecv=rank+1;
			if(npeRecv==size) npeRecv=0;
		}
		MPI_Isend(tempSendIdBuf, multMI, MPI_INT, npeSend, 1,MPI_COMM_WORLD, &req2);
		MPI_Isend(tempSendBuf, multMI*comlsqr.nAstroPSolved, MPI_DOUBLE, npeSend, 2, MPI_COMM_WORLD, &req3);

		MPI_Recv(tempRecvIdBuf, multMI, MPI_INT, npeRecv, 1,MPI_COMM_WORLD, &status);
		MPI_Recv(tempRecvBuf, multMI*comlsqr.nAstroPSolved, MPI_DOUBLE, npeRecv, 2, MPI_COMM_WORLD, &status);

		MPI_Wait(&req2,&status);
		MPI_Wait(&req3,&status);

		MPI_Barrier(MPI_COMM_WORLD);					
		
		
		int okupd=0;
		if(tempRecvIdBuf[1]==comlsqr.mapStar[rank][0])
		{
		   for(int ns=0;ns<comlsqr.nAstroPSolved;ns++)
					vectToSum[ns]+=tempRecvBuf[comlsqr.nAstroPSolved+ns];
		   okupd=1;
		}
		if(tempRecvIdBuf[1]==comlsqr.mapStar[rank][1] && okupd==0)
		{
		   for(int ns=0;ns<comlsqr.nAstroPSolved;ns++)
					vectToSum[(comlsqr.VrIdAstroPDim-1)*comlsqr.nAstroPSolved+ns]+=tempRecvBuf[comlsqr.nAstroPSolved+ns];
		}
		
	    okupd=0;
		
		if(tempRecvIdBuf[0]!=tempRecvIdBuf[1])
		{
		
		  if(tempRecvIdBuf[0]==comlsqr.mapStar[rank][1] )
 		  {
			for(int ns=0;ns<comlsqr.nAstroPSolved;ns++)
				vectToSum[(comlsqr.VrIdAstroPDim-1)*comlsqr.nAstroPSolved+ns]+=tempRecvBuf[ns];
			okupd=1;
		}
		if(tempRecvIdBuf[0]==comlsqr.mapStar[rank][0])
		{
		   for(int ns=0;ns<comlsqr.nAstroPSolved;ns++)
					vectToSum[ns]+=tempRecvBuf[ns];
		}
		
		}
		
	} 
		
		

    free(tempSendBuf); 
    free(tempRecvBuf);


}



void initThread(struct comData *comlsqr)
{
int myid=comlsqr->myid;

comlsqr->nthreads=1; 



int nthreads=comlsqr->nthreads;
/// Prepare the structure for the division of the for cycle in aprod mode=2
comlsqr->mapForThread=(long **) calloc(nthreads,sizeof(long *));
for(int n=0;n<nthreads;n++)
	comlsqr->mapForThread[n]=(long *) calloc(3,sizeof(long));

int nElements=comlsqr->mapNoss[myid]/nthreads;
comlsqr->mapForThread[0][0]=0;
comlsqr->mapForThread[0][1]=nElements/2;
comlsqr->mapForThread[0][2]=nElements;
if(comlsqr->mapNoss[myid]%nthreads>0)  comlsqr->mapForThread[0][2]++;

for(int n=1;n<nthreads;n++)
{
	comlsqr->mapForThread[n][0]=comlsqr->mapForThread[n-1][2];
	comlsqr->mapForThread[n][1]=comlsqr->mapForThread[n][0]+nElements/2;
	comlsqr->mapForThread[n][2]=comlsqr->mapForThread[n][0]+nElements;
	if(comlsqr->mapNoss[myid]%nthreads>n)  comlsqr->mapForThread[n][2]++;
}
comlsqr->mapForThread[nthreads-1][2]=comlsqr->mapNoss[myid];
		
/////
if(comlsqr->myid==0) printf("\n\nRunning with OMP: nthreads=%d\n\n",nthreads); 

}




// This function computes the product of system matrix by precondVect. This avoids to compute the produsct in aprod for each iteration.
void precondSystemMatrix(double *sysmatAstro,double *sysmatAtt,double *sysmatInstr,double *sysmatGloB,double *sysmatConstr, double *preCondVect, long *matrixIndexAstro,long  *matrixIndexAtt,int *instrCol,struct comData comlsqr)
{

    int myid;
    long int *mapNoss;
    long int j, l=0;
    int ii;
    int setBound[4];
  
    myid=comlsqr.myid;
    mapNoss=comlsqr.mapNoss;
    
    short nAstroPSolved=comlsqr.nAstroPSolved;
    short nInstrPSolved=comlsqr.nInstrPSolved;
    long nparam=comlsqr.parOss;
    short nAttParAxis=comlsqr.nAttParAxis;
    long nDegFredoomAtt=comlsqr.nDegFreedomAtt;
    long VrIdAstroPDimMax=comlsqr.VrIdAstroPDimMax;
    long offsetAttParam=comlsqr.offsetAttParam;
    long offsetInstrParam=comlsqr.offsetInstrParam;
    long offsetGlobParam=comlsqr.offsetGlobParam;
    int extConstraint=comlsqr.extConstraint;
    int nEqExtConstr=comlsqr.nEqExtConstr;
    int numOfExtStar=comlsqr.numOfExtStar;
    int barConstraint=comlsqr.barConstraint;
    int nEqBarConstr=comlsqr.nEqBarConstr;
    int numOfBarStar=comlsqr.numOfBarStar;
    int numOfExtAttCol=comlsqr.numOfExtAttCol;
    int startingAttColExtConstr=comlsqr.startingAttColExtConstr;
    short nAttAxes=comlsqr.nAttAxes;
    int nElemIC=comlsqr.nElemIC;
    long VroffsetAttParam=comlsqr.VroffsetAttParam;
    
    setBound[0]=comlsqr.setBound[0];
    setBound[1]=comlsqr.setBound[1];
    setBound[2]=comlsqr.setBound[2];
    setBound[3]=comlsqr.setBound[3];

    for(long i=0;i<comlsqr.mapNoss[myid];i++){
        for(ii=0;ii<nAstroPSolved;ii++){
            if(ii==0){
                long numOfStarPos=matrixIndexAstro[i]/nAstroPSolved;
                j=(numOfStarPos-comlsqr.mapStar[myid][0])*nAstroPSolved;
            }else{
                ++j;
            }
            sysmatAstro[i*nAstroPSolved+ii]=sysmatAstro[i*nAstroPSolved+ii]*preCondVect[j];
        }
    }

    for(long i=0;i<comlsqr.mapNoss[myid];i++){
        long counterAxis=0;
        for(ii=0;ii<comlsqr.nAttP;ii++){
            if((ii % nAttParAxis)==0) {
                j=matrixIndexAtt[i]+counterAxis*nDegFredoomAtt+(VrIdAstroPDimMax*nAstroPSolved-offsetAttParam);
                counterAxis++;
            }else{
                j++;
            }
            sysmatAtt[i*comlsqr.nAttP+ii]=sysmatAtt[i*comlsqr.nAttP+ii]*preCondVect[j];
        }
    }

    for(long i=0;i<comlsqr.mapNoss[myid];i++){
        long counterInstr=0;
        for(ii=0;ii<nInstrPSolved;ii++){
            j=offsetInstrParam+instrCol[i*nInstrPSolved+counterInstr]+(VrIdAstroPDimMax*nAstroPSolved-offsetAttParam);
            counterInstr++;
            sysmatInstr[i*nInstrPSolved+ii]=sysmatInstr[i*nInstrPSolved+ii]*preCondVect[j];
        }
    }

    for(long i=0;i<comlsqr.mapNoss[myid];i++){
        for(ii=0;ii<comlsqr.nGlobP;ii++){
            if(ii==0){
                j=offsetGlobParam+(VrIdAstroPDimMax*nAstroPSolved-offsetAttParam);
            }else{
                j++;
            }
            sysmatGloB[i*comlsqr.nGlobP+ii]=sysmatGloB[i*comlsqr.nGlobP+ii]*preCondVect[j];
        }
    }

    l=0;
    if(extConstraint){
        for(int i=0;i<nEqExtConstr;i++){
            for(int ns=0;ns<nAstroPSolved*numOfExtStar;ns++){
                sysmatConstr[l]=sysmatConstr[l]*preCondVect[ns];
                l++;
            }
            for(int naxis=0;naxis<nAttAxes;naxis++){
                for(int j=0;j<numOfExtAttCol;j++){
                    int ncolumn = VrIdAstroPDimMax*nAstroPSolved+startingAttColExtConstr+j+naxis*nDegFredoomAtt;
                    sysmatConstr[l]=sysmatConstr[l]*preCondVect[ncolumn];
                    l++;
                }
            }

        }
    }
    if(barConstraint){
        for(int i=0;i<nEqBarConstr;i++){
            for(int ns=0;ns<nAstroPSolved*numOfBarStar;ns++){
                sysmatConstr[l]=sysmatConstr[l]*preCondVect[ns];
                l++;
            }
            
        }
    }
    if(nElemIC>0){
        for(int i=0;i<nElemIC;i++){
            int ncolumn=offsetInstrParam+(VroffsetAttParam-offsetAttParam)+instrCol[mapNoss[myid]*nInstrPSolved+i];
            sysmatConstr[l]=sysmatConstr[l]*preCondVect[ncolumn];
            l++;
        }
    }



  

}    


/* Generates a pseudo-random number having a gaussian distribution
 * with mean ave e rms sigma.
 * The init2 parameter is used only when the the pseudo-random
 * extractor is the ran2() from Numerical Recipes instead of the
 * standard rand() system function.
 */
double gauss(double ave, double sigma, long init2)
{
    int i;
    double rnd;
    
    rnd=0.0;
    for(i=1; i<=12; i++)
	// comment the following line and uncomment the next one
	// to use the system rountine for random numbers
	rnd += ran2(&init2);
    // rnd += ((double) rand()/RAND_MAX);
    rnd -= 6.0;
    rnd = ave+sigma*rnd;
    
    return rnd;
    
}

/* From "Numerical Recipes in C". Generates random numbers.
 * Requires a pointer to long as seed and gives a double as the result.
 */
double ran2(long *idum)
/* Long period (> 2 . 10 18 ) random number generator of L'Ecuyer with
   Bays-Durham shu.e and added safeguards. Returns a uniform random deviate
   between 0.0 and 1.0 (exclusive of the endpoint values). Call with idum a
   negative integer to initialize; thereafter, do not alter idum between
   successive deviates in a sequence. RNMX should approximate the largest
   oating value that is less than 1.
   */
{
   int j;
   long k;
   static long idum2=123456789;
   static long iy=0;
   static long iv[NTAB];
   double temp;
   
   if (*idum <= 0) {                 // Initialize.
       if (-(*idum) < 1) *idum=1;     // Be sure to prevent idum = 0.
       else *idum = -(*idum);
       idum2=(*idum);
       for (j=NTAB+7;j>=0;j--) {      // Load the shu.e table (after 8 warm-ups).
           k=(*idum)/IQ1;
           *idum=IA1*(*idum-k*IQ1)-k*IR1;
           if (*idum < 0) *idum += IM1;
           if (j < NTAB) iv[j] = *idum;
       }
       iy=iv[0];
   }
   k=(*idum)/IQ1;                    // Start here when not initializing.
   *idum=IA1*(*idum-k*IQ1)-k*IR1;    // Compute idum=(IA1*idum) % IM1 without
                                     // over ows by Schrage's method.
   if (*idum < 0) *idum += IM1;
   k=idum2/IQ2;
   idum2=IA2*(idum2-k*IQ2)-k*IR2;    // Compute idum2=(IA2*idum) % IM2 likewise.
   if (idum2 < 0) idum2 += IM2;
   j=iy/NDIV;                        // Will be in the range 0..NTAB-1.
   iy=iv[j]-idum2;                   // Here idum is shu.ed, idum and idum2 are
                                     // combined to generate output.
   iv[j] = *idum;
   if (iy < 1) iy += IMM1;
   if ((temp=AM*iy) > RNMX) return RNMX;  // Because users don't expect endpoint values.
   else return temp;
}



struct nullSpace cknullSpace(double *sysmatAstro,double *sysmatAtt,double *sysmatInstr,double *sysmatGloB,double *sysmatConstr,long * matrixIndexAstro,long * matrixIndexAtt,double *attNS,struct comData  comlsqr){
    struct nullSpace results;
    int nproc, myid;
    long nunkSplitNS;
    double * nullSpaceVect;
    double * prodNS;
    long nElements, nStar;
    int nEqExtConstr;
    int nparam;
    int firstStarConstr,lastStarConstr;
    int nOfElextObs;
    short nAstroPSolved,nAttParAxis;
    double *nullSpaceFPN;
    double *productNS;
    int npeSend,npeRecv;
    double sum,extConstrW;
    long sumVer;
    int setBound[4];
    long int l1, j;
    long int nDegFreedomAtt,localAstroMax,offsetAttParam;
    long int *mapNoss;
    time_t seconds[2], tot_sec;

    
    MPI_Status status;
    MPI_Request req1;

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    
    nEqExtConstr=comlsqr.nEqExtConstr;
    firstStarConstr=comlsqr.firstStarConstr;
    lastStarConstr=comlsqr.lastStarConstr;
    nAstroPSolved=comlsqr.nAstroPSolved;
    nOfElextObs=comlsqr.nOfElextObs;
    nDegFreedomAtt=comlsqr.nDegFreedomAtt;
    nAttParAxis=comlsqr.nAttParAxis;
    localAstroMax=comlsqr.VrIdAstroPDimMax*nAstroPSolved;
    offsetAttParam=comlsqr.offsetAttParam;
    mapNoss=comlsqr.mapNoss;
    setBound[0]=comlsqr.setBound[0];
    setBound[1]=comlsqr.setBound[1];
    setBound[2]=comlsqr.setBound[2];
    setBound[3]=comlsqr.setBound[3];
    extConstrW=comlsqr.extConstrW;
    nStar=comlsqr.nStar;
    int nAttParam=comlsqr.nAttParam;
    int nInstrParam=comlsqr.nInstrParam;
    int nGlobalParam=comlsqr.nGlobalParam;
    
    nparam=nAstroPSolved+comlsqr.nAttP+comlsqr.nInstrPSolved+comlsqr.nGlobP;
    
    nElements = mapNoss[myid]+nEqExtConstr;
    
    nullSpaceFPN = (double *) calloc(nAstroPSolved*nEqExtConstr, sizeof(double));
    if (!nullSpaceFPN)
        exit(err_malloc("nullSpaceFPN",myid));
     
    productNS = (double *) calloc(nElements, sizeof(double));
    if (!productNS)
        exit(err_malloc("productNS",myid));

     
       
    npeSend=myid-1;
    npeRecv=myid+1;
        
    for(int i=0; i<nEqExtConstr;i++){
        if(myid>0){
            MPI_Isend(&sysmatConstr[nOfElextObs*i],nAstroPSolved, MPI_DOUBLE, npeSend, 1,MPI_COMM_WORLD, &req1);

        }
        if(myid<nproc-1){
            MPI_Recv(&nullSpaceFPN[nAstroPSolved*i], nAstroPSolved, MPI_DOUBLE, npeRecv, 1,MPI_COMM_WORLD, &status);
        }
         if(myid>0) MPI_Wait(&req1,&status);
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
        
        
        
    prodNS = (double *) calloc(nElements, sizeof(double));
    if (!prodNS)
            exit(err_malloc("prodNS",myid));

    
    
    nunkSplitNS=localAstroMax + nAttParam+nInstrParam+nGlobalParam;
    
   
    nullSpaceVect = (double *) calloc(nunkSplitNS, sizeof(double));
    if (!nullSpaceVect) exit(err_malloc("nullSpaceVect",myid));
    
    for(int ic=0; ic<nEqExtConstr;ic++){
        seconds[0]=time(NULL);
        for(int j1=localAstroMax;j1<nunkSplitNS;j1++)
            nullSpaceVect[j1]=0.0;
            for(int j1=0;j1<nElements;j1++)
                productNS[j1]=0.0;
        for(int j1=0;j1<(lastStarConstr-firstStarConstr+1)*nAstroPSolved;j1++){
            nullSpaceVect[j1]= sysmatConstr[nOfElextObs*ic+j1]/extConstrW;
        }
        if(comlsqr.mapStar[myid][1]>lastStarConstr){
            for(int j=0;j<nAstroPSolved;j++)
                nullSpaceVect[(lastStarConstr-firstStarConstr+1)*nAstroPSolved+j]=nullSpaceFPN[ic*nAstroPSolved+j]/extConstrW;
        }

        if(ic<3)
        {
            for (int m=0;m<nDegFreedomAtt;m++)
                    nullSpaceVect[localAstroMax+ic*nDegFreedomAtt+m]=comlsqr.nullSpaceAttfact/extConstrW;
        } else{
            for (int m=0;m<nDegFreedomAtt;m++)
                nullSpaceVect[localAstroMax+(ic-3)*nDegFreedomAtt+m]=attNS[m]/extConstrW;
            
        }
{

    for(long i=0;i<mapNoss[myid];i++){
            sum=0.0;
            sumVer=0;
            int lset=0;

            long chkSumVer=(nAstroPSolved*nAstroPSolved*(ic*nStar+matrixIndexAstro[i]/nAstroPSolved)+(nAstroPSolved*(nAstroPSolved-1))/2)+4*(ic*nDegFreedomAtt+matrixIndexAtt[i]-nStar*nAstroPSolved)+6;
            
            double NSVal;

            for(int l=nAstroPSolved*i;l<nAstroPSolved*i+nAstroPSolved;++l){
                if(lset==0){
                    long numOfStarPos=matrixIndexAstro[i]/nAstroPSolved;
                    j=(numOfStarPos-comlsqr.mapStar[myid][0])*nAstroPSolved; 
                }else{
                    j++;
                }
                sum=sum+sysmatAstro[l]*nullSpaceVect[j];

                NSVal=(matrixIndexAstro[i]+lset)+ic*nStar*nAstroPSolved;
                sumVer=sumVer+1.0*NSVal;
                lset++;
            }

            for(int l=comlsqr.nAttP*i;l<comlsqr.nAttP*i+comlsqr.nAttP;++l){
                if(((lset-setBound[1]) % nAttParAxis)==0) {
                    j=matrixIndexAtt[i]+((lset-setBound[1])/nAttParAxis)*nDegFreedomAtt+(localAstroMax-offsetAttParam);
                }else{
                    j++;
                }
                sum=sum+sysmatAtt[l]*nullSpaceVect[j];

                if(ic==0 || ic==3){
                    
                    if(lset<setBound[1]+4)
                        NSVal=(matrixIndexAtt[i]-nStar*nAstroPSolved)+lset-setBound[1]+ic*nDegFreedomAtt;
                    else
                        NSVal=0;
                    }
                    
                if(ic==1 || ic==4){
                    if(lset>=setBound[1]+4 && lset<setBound[1]+8)
                        NSVal=(matrixIndexAtt[i]-nStar*nAstroPSolved)+ic*nDegFreedomAtt+lset-(setBound[1]+4);
                    else
                        NSVal=0;
                }
                if(ic==2 || ic==5){
                    if(lset>=setBound[1]+8)
                        NSVal=(matrixIndexAtt[i]-nStar*nAstroPSolved)+ic*nDegFreedomAtt+lset-(setBound[1]+8);
                    else
                        NSVal=0;
                }
                sumVer=sumVer+1.0*NSVal;
                lset++;
            }

        if(sumVer != chkSumVer){
            printf("ERROR: PE=%d NullSapce Equation ic=%d, sumVer[%d]=%ld and chkSumVer=%ld are not equal\n",myid,ic,i,sumVer,    chkSumVer);
            MPI_Abort(MPI_COMM_WORLD, 0);
            exit(1);
        }
        productNS[i]=sum;
    }


}
        double normLoc;
        normLoc=cblas_dnrm2(mapNoss[myid],productNS,1);
        double normLoc2=normLoc*normLoc;
        double nrmGlob;
        MPI_Allreduce(&normLoc2, &nrmGlob,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        results.vectNorm[ic]=sqrt(nrmGlob);
        double localMin=productNS[0];
        double localMax=productNS[0];
        double globalSum=0.0;
        double localSum=0.0;
        for(int j=0;j<comlsqr.mapNoss[myid];j++){
           if(localMin>productNS[j]) localMin=productNS[j];
            if(localMax<productNS[j]) localMax=productNS[j];
            localSum+=productNS[j];
        }
       MPI_Allreduce(&localMin,&results.compMin[ic], 1, MPI_DOUBLE, MPI_MIN,MPI_COMM_WORLD );
       MPI_Allreduce(&localMax,&results.compMax[ic], 1, MPI_DOUBLE, MPI_MAX,MPI_COMM_WORLD );
       MPI_Allreduce(&localSum,&globalSum, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD );
       double avg=globalSum/comlsqr.nobs;
       results.compAvg[ic]=avg;

       double localsqrsum=0;
       double globalsqrsum=0;
       for(int j=0;j<comlsqr.mapNoss[myid];j++)
               localsqrsum+=productNS[j]*productNS[j];
               
        MPI_Allreduce(&localsqrsum,&globalsqrsum, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD );
        results.compVar[ic]=sqrt((globalsqrsum/comlsqr.nobs)-avg*avg);
 
    } 
    free(nullSpaceFPN);
    free(productNS);
    free(prodNS);
    free(nullSpaceVect);
    return results;
}



/*--------------------------------------------------------------------------*/
// Compute the value of the shifted Legendre polinomial of degree deg. Here we
// just need deg<3.
double legendre(int deg, double x) {
	double res;
	
	switch(deg) {
	case 0:
		res = 1.0;
		break;
	case 1:
		res = 2.0*(x-0.5);
		break;
	case 2:
		res = 6.0*(x-0.5)*(x-0.5)-0.5;
		break;
	default:
		res = -1.0;
		break;
}
	
	return res;
}

int computeInstrConstr (struct comData comlsqr,double * instrCoeffConstr,int * instrColsConstr,int * instrConstrIlung)
{
    ////////////////// Writing instrConstrRows_xxxx.bin file
// There are 1 AL + 2 AC constraint equations for each time interval for the large scale parameters (total=3*nTimeIntervals)
// There are 1 AL + 1 AC constraint equations for each CCD and Legendre polinomial degree (total=2*nCCDs*3)
// The equations are described by three arrays: instrCoeffConstr, instrColsConstr, instrConstrIlung, which contain the
// coefficients, the column indexes and the length of the non-zero coefficients of each equation respectively.
// MUST BE ALLOCATED THE FOLLOWING VECTORS:
//instrCoeffConstr=(double *) calloc(nElemIC, sizeof(double));
//instrColsConstr=(int *) calloc(nElemIC, sizeof(int));
//instrConstrIlung=(int *) calloc(nOfInstrConstr, sizeof(int));

    int lsInstrFlag=comlsqr.lsInstrFlag;
    int ssInstrFlag=comlsqr.ssInstrFlag;
    long nFoVs=1+comlsqr.instrConst[0];
    long nCCDs=comlsqr.instrConst[1];
    long nPixelColumns=comlsqr.instrConst[2];
    long nTimeIntervals=comlsqr.instrConst[3];
    
int nElemICLSAL =comlsqr.nElemICLSAL;
int nElemICLSAC =comlsqr.nElemICLSAC;
int nElemICSS = comlsqr.nElemICSS;
int nOfInstrConstr = comlsqr.nOfInstrConstr;
int nElemIC = comlsqr.nElemIC;
int counterElem=0;
int counterEqs=0;
int elemAcc=0;

if(lsInstrFlag){
    // generate large scale constraint eq. AL
    for(int i=0; i<nTimeIntervals; i++) {
        instrConstrIlung[counterEqs] = nElemICLSAL;
        elemAcc+=nElemICLSAL;
        counterEqs++;
        for(int j=0; j<nFoVs; j++) {
            for(int k=0; k<nCCDs; k++) {
                instrCoeffConstr[counterElem] = 1.0;
                instrColsConstr[counterElem] = comlsqr.offsetCdelta_eta + j*nCCDs*nTimeIntervals+k*nTimeIntervals+i;
                counterElem++;
            }
        }
    }
    // generate large scale constraint eq. AC
    for(int i=0; i<nTimeIntervals; i++) {
        for(int j=0; j<nFoVs; j++) {
            instrConstrIlung[counterEqs] = nElemICLSAC;
            elemAcc+=nElemICLSAC;
            counterEqs++;
            for(int k=0; k<nCCDs; k++) {
                instrCoeffConstr[counterElem] = 1.0;
                instrColsConstr[counterElem] = comlsqr.offsetCdelta_zeta + j*nCCDs*nTimeIntervals+k*nTimeIntervals+i;
                counterElem++;
            }
        }
    }
    if(ssInstrFlag){
        // generate small scale constraint eq. AL
		double x;
		for(int i=0; i<nCCDs; i++) {
            for(int j=0; j<3; j++) { // each CCD generates 3 constraint equations, one for each order of the legendre polinomials
                instrConstrIlung[counterEqs] = nElemICSS;
                elemAcc+=nElemICSS;
                counterEqs++;
                for(int k=0; k<nPixelColumns; k++) {
					x=(k+0.5)/nPixelColumns;
                    instrCoeffConstr[counterElem] = legendre(j,x);
					if(instrCoeffConstr[counterElem]==-1.0) {
						printf("Error from legendre function when i=%d, j=%d, k=%d\n", i, j, k);
						return 0;
					}
                    instrColsConstr[counterElem] = comlsqr.offsetCnu + i*nPixelColumns + k;
                    counterElem++;
                }
            }
        }
        // generate small scale constraint eq. AC
        for(int i=0; i<nCCDs; i++) {
            for(int j=0; j<3; j++) { // each CCD generates 3 constraint equations, one for each order of the legendre polinomials
                instrConstrIlung[counterEqs] = nElemICSS;
                elemAcc+=nElemICSS;
                counterEqs++;
                for(int k=0; k<nPixelColumns; k++) {
					x=(k+0.5)/nPixelColumns;
                    instrCoeffConstr[counterElem] = legendre(j,x);
					if(instrCoeffConstr[counterElem]==-1.0) {
						printf("Error from legendre function when i=%d, j=%d, k=%d\n", i, j, k);
						return 0;
					}
                    instrColsConstr[counterElem] = comlsqr.offsetCDelta_eta_3 + i*nPixelColumns + k;
                    counterElem++;
                }
            }
        }
    }
}
if(counterEqs!=nOfInstrConstr) {
    printf("SEVERE ERROR  counterEqs =%d does not coincide with nOfInstrConstr=%d\n", counterEqs, nOfInstrConstr);
    return 0;
}
if(counterElem!=nElemIC) {
    printf("SEVERE ERROR  counterElem =%d does not coincide with nElemIC=%d\n", counterElem, nElemIC);
    return 0;
}
if(elemAcc!=nElemIC) {
    printf("SEVERE ERROR   elemAcc =%d does not coincide with nElemIC=%d\n", elemAcc, nElemIC);
    return 0;
}
    return 1;
}



float simfullram(long* nStar, long* nobs, float memGlobal, int nparam, int nAttParam, int nInstrParam){
    float smGB=0., ktGB=0., miGB=0.,iiGB=0.,auxGB=0., memGB=0., prevmemGB;
    long prevnStar, prevnobs;
    long gigaByte=1024*1024*1024;
    long ncoeff;

        
        ncoeff = nparam * *nobs; // total number of non-zero coefficients of the system
        smGB=(float)(ncoeff)*8/(gigaByte);  //systemMatrix
        ktGB=(float)(*nobs)*8/(gigaByte);     //knownTerms
        miGB=(float)(*nobs*2)*8/(gigaByte);   //matrixIndex
        iiGB=(float)(*nobs*6)*4/(gigaByte);   //InstrIndex
        auxGB=(float)(*nStar*5+nAttParam+nInstrParam+0)*8/(gigaByte); //precondVect+vVect+wVect+xSolution+standardError
        memGB=smGB+miGB+ktGB+iiGB+5*auxGB;
        if(memGlobal < memGB){
            return memGlobal;
        }
        
        while(memGB < memGlobal){
            prevnStar=*nStar;
            prevnobs=*nobs;
            prevmemGB=memGB;
            *nStar*=2;
            *nobs*=3;
            ncoeff = nparam * *nobs;
            smGB=(float)(ncoeff)*8/(gigaByte);  //systemMatrix
            ktGB=(float)(*nobs)*8/(gigaByte);     //knownTerms
            miGB=(float)(*nobs*2)*8/(gigaByte);   //matrixIndex
            iiGB=(float)(*nobs*6)*4/(gigaByte);   //InstrIndex
            auxGB=(float)(*nStar*5+nAttParam+nInstrParam+0)*8/(gigaByte); //precondVect+vVect+wVect+xSolution+standardError
            memGB=smGB+miGB+ktGB+iiGB+5*auxGB;
        }
    *nStar=prevnStar;
    *nobs=prevnobs;
    memGB=prevmemGB;
    while(memGB < memGlobal){
        prevnStar=*nStar;
        prevnobs=*nobs;
        prevmemGB=memGB;
        *nobs+=10000;
        ncoeff = nparam * *nobs;
        smGB=(float)(ncoeff)*8/(gigaByte);  //systemMatrix
        ktGB=(float)(*nobs)*8/(gigaByte);     //knownTerms
        miGB=(float)(*nobs*2)*8/(gigaByte);   //matrixIndex
        iiGB=(float)(*nobs*6)*4/(gigaByte);   //InstrIndex
        auxGB=(float)(*nStar*5+nAttParam+nInstrParam+0)*8/(gigaByte); //precondVect+vVect+wVect+xSolution+standardError
        memGB=smGB+miGB+ktGB+iiGB+5*auxGB;
    }
    *nStar=prevnStar;
    *nobs=prevnobs;
    memGB=prevmemGB;

    return prevmemGB;
}

