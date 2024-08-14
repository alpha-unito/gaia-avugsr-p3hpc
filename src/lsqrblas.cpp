/* bccblas.c
   $Revision: 231 $ $Date: 2006-04-15 18:47:05 -0700 (Sat, 15 Apr 2006) $

   ----------------------------------------------------------------------
   This file is part of BCLS (Bound-Constrained Least Squares).

   Copyright (C) 2006 Michael P. Friedlander, Department of Computer
   Science, University of British Columbia, Canada. All rights
   reserved. E-mail: <mpf@cs.ubc.ca>.
   
   BCLS is free software; you can redistribute it and/or modify it
   under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.
   
   BCLS is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General
   Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with BCLS; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301
   USA
   ----------------------------------------------------------------------
*/
#include <stdio.h>
#include <stdlib.h>
/*!
   \file

   This file contains C-wrappers to the BLAS (Basic Linear Algebra
   Subprograms) routines that are used by BCLS.  Whenever possible,
   they should be replaced by corresponding BLAS routines that have
   been optimized to the machine being used.

   Included BLAS routines:

   - cblas_daxpy
   - cblas_dcopy
   - cblas_ddot
   - cblas_dnrm2
   - cblas_dscal
*/

#include "cblas.h"


void
cblas_daxpy( const int N, const double alpha, const double *X,
             const int incX, double *Y, const int incY)
{
  int i;

  if (N     <= 0  ) return;
  if (alpha == 0.0) return;

  if (incX == 1 && incY == 1) {
      const int m = N % 4;

      for (i = 0; i < m; i++)
          Y[i] += alpha * X[i];
      
      for (i = m; i + 3 < N; i += 4) {
          Y[i    ] += alpha * X[i    ];
          Y[i + 1] += alpha * X[i + 1];
          Y[i + 2] += alpha * X[i + 2];
          Y[i + 3] += alpha * X[i + 3];
      }
  } else {
      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);

      for (i = 0; i < N; i++) {
          Y[iy] += alpha * X[ix];
          ix    += incX;
          iy    += incY;
      }
  }
}


void
cblas_dcopy( const long int N, const double *X,
             const int incX, double *Y, const int incY)
{
  long int i,ix,iy;
    if (incX > 0) ix=0;
    else ix=((N) - 1) * (-(incX));

    if (incY > 0) iy=0;
    else iy=((N) - 1) * (-(incY));


  #pragma omp parallel
  {
      long ixThread,iyThread;
      long ixThreadOffset=ix, iyThreadOffset=iy;
      #pragma omp for
      for (i = 0; i < N; i++) {
          ixThread=ixThreadOffset+incX*i;
          iyThread=iyThreadOffset+incY*i;
          Y[iyThread]  = X[ixThread];
      }
  }
}


double
cblas_ddot( const int N, const double *X,
            const int incX, const double *Y, const int incY)
{
  double r  = 0.0;
  int    i;
  int    ix = OFFSET(N, incX);
  int    iy = OFFSET(N, incY);

  for (i = 0; i < N; i++) {
      r  += X[ix] * Y[iy];
      ix += incX;
      iy += incY;
  }
  
  return r;
}



double
cblas_dnrm2( const long int N, const double *X, const int incX) 
{
  double
      scale = 0.0,
      ssq   = 1.0;
  long int
      i,
      ix    = 0;

  if (N <= 0 || incX <= 0) return 0;
  else if (N == 1)         return fabs(X[0]);

  for (i = 0; i < N; i++) {
      const double x = X[ix];

      if (x != 0.0) {
          const double ax = fabs(x);

          if (scale < ax) {
              ssq   = 1.0 + ssq * (scale / ax) * (scale / ax);
              scale = ax;
          } else {
              ssq += (ax / scale) * (ax / scale);
          }
      }

      ix += incX;
  }

  return scale * sqrt(ssq);
}


double
cblas_dnrm21( const long int N, const double *X, const int incX) 
{
  double
       ssq[100], ssqFinal=0;
  long ix;
  for (int j=0;j<100;j++) ssq[j]=0;	

  int tid=0,nthreads=0;
  

  if (N <= 0 || incX <= 0) return 0;
  else if (N == 1)         return fabs(X[0]);
  #pragma omp parallel private(tid,nthreads,ix) 
  {
    #ifdef OMP
      tid = omp_get_thread_num();
            nthreads = omp_get_num_threads();
            if(nthreads>100) exit(1);
    #endif
    
      
    #pragma omp for
      for (long i = 0; i < N; i++) {
          ix=incX*i;
          ssq[tid] += X[ix]*X[ix];
      }
  }  

  for(int j=0;j<100;j++)
    ssqFinal+=ssq[j];

  return sqrt(ssqFinal);
}


double
cblas_dnrm22( const long int N, const double *X, const int incX) 
{
  double
      scale[100],
      ssq[100],
      resultSqr[100];
  long int
      i,
      ix    = 0;
  for (int j=0;j<100;j++) {ssq[j]=1.0; scale[j]=0.0;resultSqr[j]=0.0;}	

  int tid=0,nthreads=0;

  if (N <= 0 || incX <= 0) return 0;
  else if (N == 1)         return fabs(X[0]);
  #pragma omp parallel private(ix)
  {
  #ifdef OMP
    tid = omp_get_thread_num();
          nthreads = omp_get_num_threads();
          if(nthreads>100) exit(1);
  #endif
  #pragma omp for
    for (i = 0; i < N; i++) {
        ix=incX*i;
        const double x = X[ix];

        if (x != 0.0) {
            const double ax = fabs(x);

            if (scale[tid] < ax) {
                ssq[tid]   = 1.0 + ssq[tid] * (scale[tid] / ax) * (scale[tid] / ax);
                scale[tid] = ax;
            } else {
                ssq[tid] += (ax / scale[tid]) * (ax / scale[tid]);
            }
        }

    }
    
    resultSqr[tid]=(scale[tid] * sqrt(ssq[tid]))*(scale[tid] * sqrt(ssq[tid]));
  }

  double resultFinal=0.0;
  for(int j=0;j<100;j++) resultFinal+=resultSqr[j];
    printf("resultFinal=%f\n",sqrt(resultFinal));

    return sqrt(resultFinal);
}


void cblas_dscal(const long int N, const double alpha, double *X, const int incX)
{
  long int i, ix;

  if (incX <= 0) return;

  if (incX > 0) ix=0;
  else ix=((N) - 1) * (-(incX));
  
  for (i = 0; i < N; i++) {
      X[ix] *= alpha;
      ix    += incX;
  }
}

void
cblas_dscal1(const long int N, const double alpha, double *X, const int incX)
{
  long int i, ix;

  if (incX <= 0) return;

  if (incX > 0) ix=0;
  else ix=((N) - 1) * (-(incX));
  
  #pragma omp parallel
  {
    long ixThread;
    long ixThreadOffset=ix;
  #pragma omp for
      for (i = 0; i < N; i++) {
        ixThread=ixThreadOffset+incX*i;
          X[ixThread] *= alpha;
      }
  } 
}
