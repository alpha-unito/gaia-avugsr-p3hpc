#include <mpi.h>
#include "namespaces.hpp"

void compute_system_preconditioning(const long *const &mapNoss, 
                                    const long *const &matrixIndexAstro, 
                                    const long *const &matrixIndexAtt, 
                                    const double *const &sysmatAstro, 
                                    const double *const &sysmatAtt, 
                                    const double *const &sysmatInstr, 
                                    const double *const &sysmatGloB, 
                                    const double *const &sysmatConstr, 
                                    int **const &mapStar, 
                                    double *const &preCondVect){

    long int ConstrIndex = 0, astroIndex = 0, attPIndex = 0, InstrIndex = 0, GlobIndex = 0;

    // precondvect astrometric part
    if (astro_params::nAstroPSolved > 0)
    {
        for (long ii = 0; ii < mapNoss[system_params::myid]; ++ii)
        {
            long int VrIdAstroPValue = matrixIndexAstro[ii] / astro_params::nAstroPSolved - mapStar[system_params::myid][0];
            if (VrIdAstroPValue == -1)
            {
                printf("PE=%d ERROR. Can't find gsrId for precondvect.\n", system_params::myid);
                MPI_Abort(MPI_COMM_WORLD, 1);
                exit(EXIT_FAILURE);
            }

            for (int ns = 0; ns < astro_params::nAstroPSolved; ns++)
            {
                long ncolumn = VrIdAstroPValue * astro_params::nAstroPSolved + ns;
                if (ncolumn >= system_params::nunkSplit || ncolumn < 0)
                {
                    printf("ERROR. PE=%d ncolumn=%ld ii=%ld matrixIndex[ii]=%ld\n", system_params::myid, ncolumn, ii,
                           matrixIndexAstro[ii]);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    exit(EXIT_FAILURE);
                }

                lsqr_arrs::preCondVect[ncolumn] += sysmatAstro[astroIndex] * sysmatAstro[astroIndex];
                ++astroIndex;

                if (lsqr_arrs::preCondVect[ncolumn] == 0.0)
                    printf("Astrometric: preCondVect[%ld]=0.0\n", ncolumn);
            }
        }
    }

    // precondvect att part
    for (long ii = 0; ii < mapNoss[system_params::myid]; ++ii)
    {
        for (int naxis = 0; naxis < att_params::nAttAxes; naxis++)
        {
            for (int ns = 0; ns < att_params::nAttParAxis; ns++)
            {
                long ncolumn = matrixIndexAtt[ii] + (att_params::VroffsetAttParam - att_params::offsetAttParam) + naxis * att_params::nDegFreedomAtt + ns;
                if (ncolumn >= system_params::nunkSplit || ncolumn < 0)
                {
                    printf("ERROR. PE=%d numOfAttPos=%ld nStar*nAstroPSolved=%ld ncolumn=%ld ns=%d naxis=%d matrixIndex[ii]=%ld\n",
                           system_params::myid, matrixIndexAtt[ii], astro_params::nStar * astro_params::nAstroPSolved, ncolumn, ns,
                           naxis, matrixIndexAtt[ii]);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    exit(EXIT_FAILURE);
                }

                lsqr_arrs::preCondVect[ncolumn] += sysmatAtt[attPIndex] * sysmatAtt[attPIndex];
                ++attPIndex;
                if (lsqr_arrs::preCondVect[ncolumn] == 0.0)
                    printf("Attitude: PE=%d preCondVect[%ld]=0.0 attPIndex=%ld systemMatrix[aIndex]=%12.9lf\n", system_params::myid,
                           ncolumn,
                           ii * att_params::nAttAxes * att_params::nAttParAxis + naxis * att_params::nAttParAxis + ns,
                           sysmatAtt[ii * att_params::nAttAxes * att_params::nAttParAxis +
                                     naxis * att_params::nAttParAxis +
                                     ns]); //   if aggiunto
            }
        }
    }

    // precondvect instr pat
    if (astro_params::nInstrPSolved > 0)
    {
        for (long ii = 0; ii < mapNoss[system_params::myid]; ++ii)
        {
            for (int ns = 0; ns < astro_params::nInstrPSolved; ns++)
            {
                long ncolumn =
                    instr_params::offsetInstrParam + (att_params::VroffsetAttParam - att_params::offsetAttParam) +
                    lsqr_arrs::instrCol[(ii)*astro_params::nInstrPSolved + ns];
                if (ncolumn >= system_params::nunkSplit || ncolumn < 0)
                {
                    printf("ERROR. PE=%d ii=%ld ", system_params::myid, ii);
                    for (int ke = 0; ke < astro_params::nInstrPSolved; ke++)
                        printf("instrCol[%ld]=%d ", ii + ke, lsqr_arrs::instrCol[ii + ke]);
                    printf("ncolumn=%ld   ns=%d\n", ncolumn, ns);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    exit(EXIT_FAILURE);
                }
                lsqr_arrs::preCondVect[ncolumn] += sysmatInstr[InstrIndex] * sysmatInstr[InstrIndex];
                ++InstrIndex;
                if (lsqr_arrs::preCondVect[ncolumn] == 0.0)
                    printf("Instrument: PE=%d preCondVect[%ld]=0.0 aIndex=%ld systemMatrix[InstrIndex]=%12.9lf\n", system_params::myid,
                           ncolumn, ii * astro_params::nInstrPSolved + ns,
                           sysmatInstr[ii * astro_params::nInstrPSolved + ns]); //   if aggiunto
            }
        }
    }

    // precondvect global part
    if (glob_params::nGlobP > 0)
    {
        for (long ii = 0; ii < mapNoss[system_params::myid]; ++ii)
        {
            for (int ns = 0; ns < glob_params::nGlobP; ns++)
            {
                long ncolumn = glob_params::offsetGlobParam + (att_params::VroffsetAttParam - att_params::offsetAttParam) + ns;
                if (ncolumn >= system_params::nunkSplit || ncolumn < 0)
                {
                    printf("ERROR. PE=%d ncolumn=%ld ii=%ld matrixIndex[ii+2]=%ld\n", system_params::myid, ncolumn, ii,
                           matrixIndexAstro[ii]);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    exit(EXIT_FAILURE);
                }

                lsqr_arrs::preCondVect[ncolumn] += sysmatGloB[GlobIndex] * sysmatGloB[GlobIndex];
                ++GlobIndex;
                if (lsqr_arrs::preCondVect[ncolumn] == 0)
                    printf("Global: preCondVect[%ld]=0.0\n", ncolumn); //   if aggiunto
            }
        }
    }

    /////  precondvect for extConstr
    if (constr_params::extConstraint)
    {
        for (int i = 0; i < constr_params::nEqExtConstr; i++)
        {
            long int numOfStarPos = 0;
            if (astro_params::nAstroPSolved > 0)
                numOfStarPos = constr_params::firstStarConstr;
            long int VrIdAstroPValue = -1; //

            VrIdAstroPValue = numOfStarPos - mapStar[system_params::myid][0];
            if (VrIdAstroPValue == -1)
            {
                printf("PE=%d ERROR. Can't find gsrId for precondvect.\n", system_params::myid);
                MPI_Abort(MPI_COMM_WORLD, 1);
                exit(EXIT_FAILURE);
            }
            for (int ns = 0; ns < astro_params::nAstroPSolved * constr_params::numOfExtStar; ns++)
            {
                long ncolumn = ns;
                ////
                if (ncolumn >= system_params::nunkSplit || ncolumn < 0)
                {
                    printf("ERROR. PE=%d ncolumn=%ld ii=%d matrixIndex[ii]=%ld\n", system_params::myid, ncolumn, i,
                           matrixIndexAstro[i]);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    exit(EXIT_FAILURE);
                }
                lsqr_arrs::preCondVect[ncolumn] += sysmatConstr[ConstrIndex] * sysmatConstr[ConstrIndex];
                if (lsqr_arrs::preCondVect[ncolumn] == 0.0)
                    printf("Astrometric: preCondVect[%ld]=0.0\n", ncolumn);
                ConstrIndex++;
            }
            //
            for (int naxis = 0; naxis < att_params::nAttAxes; naxis++)
            {
                for (int j = 0; j < att_params::numOfExtAttCol; j++)
                {
                    long ncolumn =
                        astro_params::VrIdAstroPDimMax * astro_params::nAstroPSolved + constr_params::startingAttColExtConstr + j +
                        naxis * att_params::nDegFreedomAtt;
                    if (ncolumn >= system_params::nunkSplit || ncolumn < 0)
                    {
                        printf("ERROR. PE=%d  ncolumn=%ld  naxis=%d j=%d\n", system_params::myid, ncolumn, naxis, j);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                        exit(EXIT_FAILURE);
                    }
                    lsqr_arrs::preCondVect[ncolumn] += sysmatConstr[ConstrIndex] * sysmatConstr[ConstrIndex];
                    ConstrIndex++;
                    if (lsqr_arrs::preCondVect[ncolumn] == 0.0)
                        printf("Attitude: PE=%d preCondVect[%ld]=0.0 aIndex=%ld systemMatrix[ConstrIndex]=%12.9lf\n",
                               system_params::myid, ncolumn, ConstrIndex, sysmatConstr[ConstrIndex]); //   if aggiunto
                }
            }
        }
    }

    ///// end precondvect for extConstr
    /////  precondvect for barConstr
    if (constr_params::barConstraint)
    {
        for (int i = 0; i < constr_params::nEqBarConstr; i++)
        {
            long int numOfStarPos = 0;
            if (astro_params::nAstroPSolved > 0)
                numOfStarPos = constr_params::firstStarConstr; // number of star associated to matrixIndex[ii]
            long int VrIdAstroPValue = -1;                     //

            VrIdAstroPValue = numOfStarPos - mapStar[system_params::myid][0];
            if (VrIdAstroPValue == -1)
            {
                printf("PE=%d ERROR. Can't find gsrId for precondvect.\n", system_params::myid);
                MPI_Abort(MPI_COMM_WORLD, 1);
                exit(EXIT_FAILURE);
            }
            for (int ns = 0; ns < astro_params::nAstroPSolved * constr_params::numOfBarStar; ns++)
            {
                long ncolumn = ns;
                ////
                if (ncolumn >= system_params::nunkSplit || ncolumn < 0)
                {
                    printf("ERROR. PE=%d ncolumn=%ld ii=%d matrixIndex[ii]=%ld\n", system_params::myid, ncolumn, i,
                           matrixIndexAstro[i]);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    exit(EXIT_FAILURE);
                }
                lsqr_arrs::preCondVect[ncolumn] += sysmatConstr[ConstrIndex] * sysmatConstr[ConstrIndex];
                ConstrIndex++;
                if (lsqr_arrs::preCondVect[ncolumn] == 0.0)
                    printf("Astrometric: preCondVect[%ld]=0.0\n", ncolumn);
            }
            //
        }
    }

    if (astro_params::nElemIC > 0)
    {
        for (int i = 0; i < astro_params::nElemIC; i++)
        {
            long ncolumn = instr_params::offsetInstrParam + (att_params::VroffsetAttParam - att_params::offsetAttParam) +
                           lsqr_arrs::instrCol[mapNoss[system_params::myid] * astro_params::nInstrPSolved + i];
            if (ncolumn >= system_params::nunkSplit || ncolumn < 0)
            {
                printf("ERROR on instrConstr. PE=%d  ncolumn=%ld   i=%d\n", system_params::myid, ncolumn, i);
                MPI_Abort(MPI_COMM_WORLD, 1);
                exit(EXIT_FAILURE);
            }
            lsqr_arrs::preCondVect[ncolumn] += sysmatConstr[ConstrIndex] * sysmatConstr[ConstrIndex];
            ConstrIndex++;
        }
    }
    ////////////////
}



    void testing_well_poseness(double *const& preCondVect,
                                const int& precond,
                                const int& myid, 
                                const long& VrIdAstroPDim, 
                                const long&VrIdAstroPDimMax,
                                const long& nAttParam,
                                const long& nInstrParam, 
                                const long& nGlobalParam, 
                                const short& nAstroPSolved){

        if (precond)
        {
            if (myid == 0)
                printf("Inverting preCondVect\n");
            for (long ii = 0; ii < VrIdAstroPDim * nAstroPSolved; ii++)
            {
                if (preCondVect[ii] == 0.0)
                {
                    if (ii - VrIdAstroPDimMax * nAstroPSolved < nAttParam)
                        printf("ERROR Att ZERO column: PE=%d preCondVect[%ld]=0.0 AttParam=%ld \n", myid, ii,
                            ii - VrIdAstroPDimMax * nAstroPSolved);
                    if (ii - VrIdAstroPDimMax * nAstroPSolved > nAttParam &&
                        ii - VrIdAstroPDimMax * nAstroPSolved < nAttParam + nInstrParam)
                        printf("ERROR Instr ZERO column: PE=%d preCondVect[%ld]=0.0 InstrParam=%ld \n", myid, ii,
                            ii - (VrIdAstroPDimMax * nAstroPSolved + nAttParam));
                    if (ii - VrIdAstroPDimMax * nAstroPSolved > nAttParam + nInstrParam)
                        printf("ERROR Global ZERO column: PE=%d preCondVect[%ld]=0.0 GlobalParam=%ld \n", myid, ii,
                            ii - (VrIdAstroPDimMax * nAstroPSolved + nAttParam + nInstrParam));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    exit(EXIT_FAILURE);
                }
                preCondVect[ii] = 1.0 / sqrt(preCondVect[ii]);
            }
            for (long ii = VrIdAstroPDimMax * nAstroPSolved;
                ii < VrIdAstroPDimMax * nAstroPSolved + nAttParam + nInstrParam + nGlobalParam; ii++)
            {
                if (preCondVect[ii] == 0.0)
                {
                    printf("ERROR non-Astrometric ZERO column: PE=%d preCondVect[%ld]=0.0\n", myid, ii);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    exit(EXIT_FAILURE);
                }
                preCondVect[ii] = 1.0 / sqrt(preCondVect[ii]);
            }
        }
        else
        {
            if (myid == 0)
                printf("Setting preCondVect to 1.0\n");
            for (long ii = 0; ii < VrIdAstroPDim * nAstroPSolved; ii++)
            {
                if (preCondVect[ii] == 0.0)
                {
                    printf("ERROR Astrometric ZERO column: PE=%d preCondVect[%ld]=0.0 Star=%ld\n", myid, ii,
                        ii / nAstroPSolved);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    exit(EXIT_FAILURE);
                }
                preCondVect[ii] = 1.0;
            }
            for (long ii = VrIdAstroPDimMax * nAstroPSolved;
                ii < VrIdAstroPDimMax * nAstroPSolved + nAttParam + nInstrParam + nGlobalParam; ii++)
            {
                if (preCondVect[ii] == 0.0)
                {
                    if (ii - VrIdAstroPDimMax * nAstroPSolved < nAttParam)
                        printf("ERROR Att ZERO column: PE=%d preCondVect[%ld]=0.0 AttParam=%ld \n", myid, ii,
                            ii - VrIdAstroPDimMax * nAstroPSolved);
                    if (ii - VrIdAstroPDimMax * nAstroPSolved > nAttParam &&
                        ii - VrIdAstroPDimMax * nAstroPSolved < nAttParam + nInstrParam)
                        printf("ERROR Instr ZERO column: PE=%d preCondVect[%ld]=0.0 InstrParam=%ld \n", myid, ii,
                            ii - (VrIdAstroPDimMax * nAstroPSolved + nAttParam));
                    if (ii - VrIdAstroPDimMax * nAstroPSolved > nAttParam + nInstrParam)
                        printf("ERROR Global ZERO column: PE=%d preCondVect[%ld]=0.0 GlobalParam=%ld \n", myid, ii,
                            ii - (VrIdAstroPDimMax * nAstroPSolved + nAttParam + nInstrParam));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    exit(EXIT_FAILURE);
                }
                preCondVect[ii] = 1.0;
            }
        }

    }



std::tuple<double,double,double> de_preconditioning(double *const& xSolution,
                        double *const& standardError,
                        const double * const& preCondVect,
                        const long& nunkSplit,
                        const long& VroffsetAttParam,
                        const long& VrIdAstroPDim,
                        const int& myid,
                        const int& idtest,
                        const short& nAstroPSolved){
        long thetaCol = 0, muthetaCol = 0, flagTheta = 0, flagMuTheta = 0;
        ldiv_t res_ldiv;

        switch (nAstroPSolved)
        {
        case 2:
            thetaCol = 1;
            muthetaCol = 0;
            break;
        case 3:
            thetaCol = 2;
            muthetaCol = 0;
            break;
        case 4:
            thetaCol = 1;
            muthetaCol = 3;
            break;
        case 5:
            thetaCol = 2;
            muthetaCol = 4;
            break;
        default:
            break;
        }

        double epsilon, localSumX = 0, localSumXsq = 0, sumX = 0, sumXsq = 0, dev = 0, localMaxDev = 0, maxDev = 0;

        ///////// Each PE runs over the its Astrometric piece
        if (muthetaCol == 0)
            for (long ii = 0; ii < VrIdAstroPDim * nAstroPSolved; ii++)
            {
                res_ldiv = ldiv((ii - thetaCol), nAstroPSolved);
                flagTheta = res_ldiv.rem;
                if (flagTheta == 0)
                {
                    xSolution[ii] *= (-preCondVect[ii]);
                    if (idtest)
                    {
                        epsilon = xSolution[ii] + 1.0;
                        localSumX -= epsilon;
                        dev = fabs(epsilon);
                        if (dev > localMaxDev)
                            localMaxDev = dev;
                    }
                }
                else
                {
                    xSolution[ii] *= preCondVect[ii]; // the corrections in theta are converted for consistency with the naming conventions in the Data Model to corrections in delta by a change of sign (Mantis Issue 0013081)
                    if (idtest)
                    {
                        epsilon = xSolution[ii] - 1.0;
                        localSumX += epsilon;
                        dev = fabs(epsilon);
                        if (dev > localMaxDev)
                            localMaxDev = dev;
                    }
                }
                if (idtest)
                    localSumXsq += epsilon * epsilon;
                standardError[ii] *= preCondVect[ii];
            }
        else
            for (long ii = 0; ii < VrIdAstroPDim * nAstroPSolved; ii++)
            {
                res_ldiv = ldiv((ii - thetaCol), nAstroPSolved);
                flagTheta = res_ldiv.rem;
                res_ldiv = ldiv((ii - muthetaCol), nAstroPSolved);
                flagMuTheta = res_ldiv.rem;
                if ((flagTheta == 0) || (flagMuTheta == 0))
                {
                    xSolution[ii] *= (-preCondVect[ii]);
                    if (idtest)
                    {
                        epsilon = xSolution[ii] + 1.0;
                        localSumX -= epsilon;
                        dev = fabs(epsilon);
                        if (dev > localMaxDev)
                            localMaxDev = dev;
                    }
                }
                else
                {
                    xSolution[ii] *= preCondVect[ii]; // the corrections in theta are converted for consistency with the naming conventions in the Data Model to corrections in delta by a change of sign (Mantis Issue 0013081)
                    if (idtest)
                    {
                        epsilon = xSolution[ii] - 1.0;
                        localSumX += epsilon;
                        dev = fabs(epsilon);
                        if (dev > localMaxDev)
                            localMaxDev = dev;
                    }
                }
                if (idtest)
                    localSumXsq += epsilon * epsilon;
                standardError[ii] *= preCondVect[ii];
            }
        //////////// End of de-preconditioning for the Astrometric unknowns

        //////////// Then only PE=0 runs over the shared unknowns (Attitude, Instrument, and Global)
        if (myid == 0)
            for (long ii = VroffsetAttParam; ii < nunkSplit; ii++)
            {
                xSolution[ii] *= preCondVect[ii];
                if (idtest)
                {
                    localSumX += (xSolution[ii] - 1.0);
                    dev = fabs(1.0 - xSolution[ii]);
                    if (dev > localMaxDev)
                        localMaxDev = dev;
                    localSumXsq += (xSolution[ii] - 1.0) * (xSolution[ii] - 1.0);
                }
                standardError[ii] *= preCondVect[ii];
            }
        //////////// End of de-preconditioning for the shared unknowns

        if (idtest)
        {
            MPI_Reduce(&localSumX, &sumX, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&localSumXsq, &sumXsq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&localMaxDev, &maxDev, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        }


        return std::make_tuple(sumX,sumXsq,maxDev);

    }