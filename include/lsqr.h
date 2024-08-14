/* lsqr.h
   $Revision: 229 $ $Date: 2006-04-15 18:40:08 -0700 (Sat, 15 Apr 2006) $
*/
/*!
   \file
   Header file for ISO C version of LSQR.
*/
#include "util.h" 

void aprod(int mode, long int m, long int n, double x[], double y[],
      double *ra,
//      long int *na,
      long int *matrixIndex,int *instrIndex,
      double *d, int *instrConstrIlung,struct comData comlsqr,time_t *ompSec);
// void lsqr( long int m,
//       long int n,
// /*      void (*aprod)(int mode, int m, int n, double x[], double y[],
//                     void *UsrWrk ),*/
//       double damp,
// //      void   *UsrWrk,
//       double u[],    // len = m
//       double v[],    // len = n
//       double w[],    // len = n
//       double x[],    // len = n
//       double se[],   // len = *
//       // double cov[],
//       double atol,
//       double btol,
//       double conlim,
//       int    itnlim,
//       // The remaining variables are output only.
//       int    *istop_out,
//       int    *itn_out,
//       double *anorm_out,
//       double *acond_out,
//       double *rnorm_out,
//       double *arnorm_out,
//       double *xnorm_out,
//       double *ra,
// //      long int *na,
//       long int *matrixIndex,
//       int *instrIndex,
//         int *instrConstrIlung,
//       // double *d, 
//       struct comData comlsqr
//        );


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
	  struct comData comlsqr);