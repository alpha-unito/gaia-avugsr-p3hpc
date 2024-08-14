/* This program solves the relativistic sphere with an
 iterative method (conjugate gradients). Dynamic case.
 It is the c translation of the original fortran program
 solvergaia.for
 M.G. Lattanzi, A. Vecchiato, B. Bucciarelli (23 May 1996)
 A. Vecchiato, R. Morbidelli for the Fortran 90 version with
 dynamical memory allocation (21 March 2005)

 A. Vecchiato, June 16, 2008

 Version 2.1 , Feb 8, 2012
 Version history:
 - version 2.1, Feb  8 2012 - Added debug mode
 - version 2.0, Jan 27 2012 - Added Checkpoint & Restart
 - version 1.2, Jan 24 2012 -
 - version 5.0 - May 2013 from Ugo Becciani and Alberto Vecchiato
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <dirent.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <tuple>

#include "lsqr.h"
#include "util.h"

#define MAX_CONSTR 1000

#include "gaiasim_includes/allocators.hpp"
#include "gaiasim_includes/namespaces.hpp"
#include "gaiasim_includes/solver_utils.hpp"
#include "gaiasim_includes/constr_generator.hpp"
#include "gaiasim_includes/precondition.hpp"
#include "gaiasim_includes/comlsqr.hpp"

/* Start of main program */
int main(int argc, char **argv)
{
    // Internal variables

    int nfileProc = 3;
    int addElementAtt = 0;
    int addElementextStar = 0;
    int addElementbarStar = 0;

    double *attNS = nullptr;

    time_t seconds[10], tot_sec[10];
    seconds[0] = time(NULL);
    double  startTime, endTime;

    int zeroAtt = 0, zeroInstr = 0, zeroGlob = 0, seqStar = 1;

    // Distributed array maps
    long int *mapNcoeff, *mapNoss;

    int itnCPR = DEFAULT_ITNCPR, itnCPRstop = 1000000;

    // serach for mapPytorchStreamReader failed locating file constants.pkl: file not found
    long lastStar = -1, firstStar = -1;

    struct comData comlsqr;
    struct nullSpace nullSpaceCk;

    long numberOfCovEle = 0;
    int covStarStarCounter = 0, covStarOtherCounter = 0, covOtherOtherCounter = 0, numberOfCompCovEle = 0;

    parse_cli(argc, argv, &zeroAtt, &zeroInstr, &zeroGlob);

    ACTION_IF_RANK_0(system_params::myid, print_cmdline_input(nfileProc, itnCPR, zeroAtt, zeroInstr, zeroGlob, numberOfCovEle))

    printf("\nProcess %d running on %s\n", system_params::myid, system_params::processor_name);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &system_params::nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &system_params::myid);
    MPI_Get_processor_name(system_params::processor_name, &system_params::namelen);
    startTime = MPI_Wtime();

    // astro_params::nAstroParam = astro_params::nStar * astro_params::nAstroPSolved;
    astro_params::nAstroParam = 0;

    att_params::nAttParam = att_params::nDegFreedomAtt * att_params::nAttAxes;
    att_params::nAttP = att_params::nAttAxes * att_params::nAttParAxis;

    long nFoVs = 1 + instr_params::instrConst[0], nCCDs = instr_params::instrConst[1], nPixelColumns = instr_params::instrConst[2], nTimeIntervals = instr_params::instrConst[3];
    instr_params::nInstrParam = instr_params::maInstrFlag * nCCDs + instr_params::nuInstrFlag * nFoVs * nCCDs +
                                instr_params::ssInstrFlag * 2 * nCCDs * nPixelColumns +
                                instr_params::lsInstrFlag * 2 * 3 * nFoVs * nCCDs * nTimeIntervals;

    glob_params::nGlobalParam = glob_params::nGlobP;

    system_params::nparam = astro_params::nAstroPSolved + att_params::nAttP + astro_params::nInstrPSolved + glob_params::nGlobP;

    print_params(numberOfCovEle);

    endTime = MPI_Wtime();
    if ((att_params::nDegFreedomAtt == 0 && att_params::nAttAxes > 0) ||
        (att_params::nDegFreedomAtt > 0 && att_params::nAttAxes == 0))
    {
        if (system_params::myid == 0)
        {
            printf("inconsistent values for nDegFreedomAtt=%ld and nAttaxes=%d\n", att_params::nDegFreedomAtt,
                   att_params::nAttAxes);
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(EXIT_FAILURE);
        }
    }

    ACTION_IF_RANK_0(system_params::myid, printf("Time to read initial parameters =%f sec.\n", endTime - startTime));

    if (astro_params::nInstrPSolved == 0)
        zeroInstr = 1;

    if (astro_params::nElemIC < 0 || astro_params::nOfInstrConstr < 0)
    {
        astro_params::nOfInstrConstrLSAL = instr_params::lsInstrFlag * nTimeIntervals;
        astro_params::nElemICLSAL = nFoVs * nCCDs;
        astro_params::nOfInstrConstrLSAC = instr_params::lsInstrFlag * nFoVs * nTimeIntervals;
        astro_params::nElemICLSAC = nCCDs;
        astro_params::nOfInstrConstrSS = instr_params::lsInstrFlag * instr_params::ssInstrFlag * 2 * nCCDs * 3;
        astro_params::nElemICSS = nPixelColumns;
        astro_params::nOfInstrConstr = astro_params::nOfInstrConstrLSAL + astro_params::nOfInstrConstrLSAC + astro_params::nOfInstrConstrSS;
        astro_params::nElemIC = astro_params::nOfInstrConstrLSAL * astro_params::nElemICLSAL +
                                astro_params::nOfInstrConstrLSAC * astro_params::nElemICLSAC +
                                astro_params::nOfInstrConstrSS * astro_params::nElemICSS;
    }


    if (astro_params::nAstroPSolved)
    {
        astro_params::mapAstroP=fast_allocate_vector<int>(astro_params::nAstroPSolved);

        switch (astro_params::nAstroPSolved)
        {
        case 2:
            astro_params::mapAstroP[0] = 1;
            astro_params::mapAstroP[1] = 2;
            if (constr_params::extConstraint)
                constr_params::nEqExtConstr = 3;
            if (constr_params::barConstraint)
                constr_params::nEqBarConstr = 3;
            break;
        case 3:
            astro_params::mapAstroP[0] = 0;
            astro_params::mapAstroP[1] = 1;
            astro_params::mapAstroP[2] = 2;
            if (constr_params::extConstraint)
                constr_params::nEqExtConstr = 3;
            if (constr_params::barConstraint)
                constr_params::nEqBarConstr = 3;
            break;
        case 4:
            for (int i = 0; i < 4; i++)
                astro_params::mapAstroP[i] = i + 1;
            break;
        case 5:
            for (int i = 0; i < 5; i++)
                astro_params::mapAstroP[i] = i;
            break;
        default:
            if (system_params::myid == 0)
            {
                printf("nAstroPSolved=%d, invalid value. Aborting.\n", astro_params::nAstroPSolved);
                MPI_Abort(MPI_COMM_WORLD, 1);
                exit(EXIT_FAILURE);
            }
        }
    }

    system_params::nunk = astro_params::nAstroParam + att_params::nAttParam + instr_params::nInstrParam + glob_params::nGlobalParam; // number of unknowns (i.e. columns of the system matrix)
    system_params::nparam = astro_params::nAstroPSolved + att_params::nAttP + astro_params::nInstrPSolved + glob_params::nGlobP;     // number of non-zero coefficients for each observation (i.e. for each system row)
    if (system_params::nparam == 0)
    {
        printf("Abort. Empty system nparam=0 . nAstroPSolved=%d nAttP=%d nInstrPSolved=%d nGlobP=%d\n",
               astro_params::nAstroPSolved, att_params::nAttP, astro_params::nInstrPSolved, glob_params::nGlobP);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (system_params::nobs <= system_params::nunk)
    {
        printf("SEVERE ERROR: number of equations=%ld and number of unknown=%ld make solution unsafe\n",
               system_params::nobs, system_params::nunk);
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(EXIT_FAILURE);
    }

    mapNoss=fast_allocate_vector<long int>(system_params::nproc);
    mapNcoeff=fast_allocate_vector<long int>(system_params::nproc);

    work_distribution(mapNoss,mapNcoeff,system_params::nobs,system_params::nproc,system_params::nparam,system_params::myid);

    ////////////////// Simulating the ... of NObsxStar file

    if (constr_params::extConstraint)
        extNobxStarfile(mapNoss);
    else if (constr_params::barConstraint)
        barNobxStarfile(mapNoss);

    //---------------------------------------------------------------------------

    init_comlsqr(&comlsqr, mapNoss, mapNcoeff, itnCPR, itnCPRstop);

    if (constr_params::extConstraint)
    {
        addElementextStar = (constr_params::lastStarConstr - constr_params::firstStarConstr + 1) * astro_params::nAstroPSolved;
        addElementAtt = att_params::numOfExtAttCol * att_params::nAttAxes;
    }
    else if (constr_params::barConstraint)
        addElementbarStar = (constr_params::lastStarConstr - constr_params::firstStarConstr + 1) * astro_params::nAstroPSolved;

    constr_params::nOfElextObs = addElementextStar + addElementAtt;
    constr_params::nOfElBarObs = addElementbarStar;
    comlsqr.nOfElextObs = constr_params::nOfElextObs;
    comlsqr.nOfElBarObs = constr_params::nOfElBarObs;

    const int totConstr = constr_params::nOfElextObs * constr_params::nEqExtConstr + constr_params::nOfElBarObs * constr_params::nEqBarConstr + astro_params::nElemIC;

    system_params::totmem = (mapNcoeff[system_params::myid] + constr_params::nOfElextObs * constr_params::nEqExtConstr + constr_params::nOfElBarObs * constr_params::nEqBarConstr + astro_params::nElemIC) * sizeof(double) + mapNoss[system_params::myid] * sizeof(long int) + astro_params::nOfInstrConstr * sizeof(int);

    system_params::nElements = mapNoss[system_params::myid];
    if (constr_params::extConstraint)       system_params::nElements += constr_params::nEqExtConstr;
    if (constr_params::barConstraint)       system_params::nElements += constr_params::nEqBarConstr;
    if (astro_params::nOfInstrConstr > 0)   system_params::nElements += astro_params::nOfInstrConstr;

    lsqr_arrs::allocate_system_matrix_knownTerms(mapNoss[system_params::myid],system_params::nElements, astro_params::nAstroPSolved, att_params::nAttP, astro_params::nInstrPSolved, glob_params::nGlobP, totConstr, system_params::myid);
    lsqr_arrs::allocate_matrix_topology(mapNoss[system_params::myid] * astro_params::nInstrPSolved + astro_params::nElemIC, astro_params::nOfInstrConstr, mapNoss[system_params::myid]);

    system_params::totmem += system_params::nElements * sizeof(double);

    att_params::offsetAttParam = astro_params::nAstroParam;
    instr_params::offsetInstrParam = att_params::offsetAttParam + att_params::nAttParam;
    glob_params::offsetGlobParam = instr_params::offsetInstrParam + instr_params::nInstrParam;
    comlsqr.offsetAttParam = att_params::offsetAttParam;


    double sysmat_start_time = MPI_Wtime();
    fill_system(lsqr_arrs::sysmatAstro, lsqr_arrs::sysmatAtt, lsqr_arrs::sysmatInstr, lsqr_arrs::sysmatGloB, mapNoss, lsqr_arrs::matrixIndexAstro, lsqr_arrs::matrixIndexAtt, zeroAtt, zeroInstr);



    if (constr_params::extConstraint) // generate extConstr on systemMatrix
        generate_external_constraints(lsqr_arrs::sysmatConstr, lsqr_arrs::knownTerms, mapNoss, attNS, addElementextStar, addElementAtt);


    if (constr_params::barConstraint) // generate barConstr on systemMatrix
        generate_bar_constraints(lsqr_arrs::sysmatConstr, lsqr_arrs::knownTerms, mapNoss, addElementbarStar);


    set_comlsqr_offsets(&comlsqr, nCCDs, nFoVs, nPixelColumns, nTimeIntervals);

    if (system_params::myid == 0 && instr_params::lsInstrFlag && astro_params::nElemIC != 0)
        generate_instr_constraints(comlsqr, lsqr_arrs::sysmatConstr, lsqr_arrs::knownTerms, mapNoss);
    

    MPI_Bcast(&lsqr_arrs::sysmatConstr[constr_params::nOfElextObs * constr_params::nEqExtConstr + constr_params::nOfElBarObs * constr_params::nEqBarConstr], astro_params::nElemIC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lsqr_arrs::instrCol[mapNoss[system_params::myid] * astro_params::nInstrPSolved], astro_params::nElemIC, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(lsqr_arrs::instrConstrIlung, astro_params::nOfInstrConstr, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lsqr_arrs::knownTerms[mapNoss[system_params::myid] + constr_params::nEqExtConstr + constr_params::nEqBarConstr], astro_params::nOfInstrConstr, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    /////// Search for map
    if (astro_params::nAstroPSolved)
    {
        firstStar = lsqr_arrs::matrixIndexAstro[0] / astro_params::nAstroPSolved;
        lastStar = lsqr_arrs::matrixIndexAstro[mapNoss[system_params::myid] - 1] / astro_params::nAstroPSolved;
        seqStar = lastStar - firstStar + 1;
    }
    else
    {
        firstStar = 0;
        lastStar = 0;
    }

    if (constr_params::extConstraint && (firstStar != constr_params::firstStarConstr || lastStar != constr_params::lastStarConstr + constr_params::starOverlap))
    {
        printf("PE=%d Error extConstraint:  firstStar=%ld firstStarConstr=%ld lastStar=%ld lastStarConstr=%ld\n", system_params::myid, firstStar, constr_params::firstStarConstr, lastStar, constr_params::lastStarConstr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(EXIT_FAILURE);
    }
    if (constr_params::barConstraint && (firstStar != constr_params::firstStarConstr || lastStar != constr_params::lastStarConstr + constr_params::starOverlap))
    {
        printf("PE=%d Error barConstraint:  firstStar=%ld firstStarConstr=%ld lastStar=%ld lastStarConstr=%ld\n", system_params::myid, firstStar, constr_params::firstStarConstr, lastStar, constr_params::lastStarConstr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(EXIT_FAILURE);
    }
    ////////////////////////////////////////////////////////////////////

    //////////// Identity solution test
    // There are two possible IDtest modes: compatible and incompatible.
    // * in "compatible" mode the known term read from the input file is substituted by
    //   the sum of the coefficients of the system row, solving in this way a compatible
    //   system whose solution is exactly a vector of ones apart from numerical approximations
    //   (a further test is done in this case, in order to check that the difference between
    //   the sum and the known term is less than 1e-12)
    // * in "incompatible" mode the known term read from the input file is substituted by
    //   the sum of the coefficients of the system row further perturbed by a quantity x
    //   extracted from a random distribution having 0 mean and a sigma determined by the
    //   variable srIDtest. In this way the problem is reconducted to the solution of an
    //   incompatible system whose solution is close to the previous one. In particular,
    //   srIDtest represents the ratio between the sigma and the unperturbed known term
    //   (i.e. the sum of the coefficients since we are in IDtest mode). Therefore the
    //   perturbed known term is computed as
    //                KT_perturbed = KT_unperturbed*(1+x)

    if (lsqr_input::idtest) // if Identity test, overwrite the value of the knownterm
    {
        for (long ii = 0; ii < mapNoss[system_params::myid]; ii++)
        {
            lsqr_arrs::knownTerms[ii] = 0.;

            for (long jj = 0; jj < astro_params::nAstroPSolved; jj++)
                lsqr_arrs::knownTerms[ii] += lsqr_arrs::sysmatAstro[ii * astro_params::nAstroPSolved + jj];
            for (long jj = 0; jj < att_params::nAttP; jj++)
                lsqr_arrs::knownTerms[ii] += lsqr_arrs::sysmatAtt[ii * att_params::nAttP + jj];
            for (long jj = 0; jj < astro_params::nInstrPSolved; jj++)
                lsqr_arrs::knownTerms[ii] += lsqr_arrs::sysmatInstr[ii * astro_params::nInstrPSolved + jj];
            for (long jj = 0; jj < glob_params::nGlobP; jj++)
                lsqr_arrs::knownTerms[ii] += lsqr_arrs::sysmatGloB[ii * glob_params::nGlobP + jj];
        }
    }
    //////////////////////////////////////
    double sysmat_end_time = MPI_Wtime() - sysmat_start_time;
    ACTION_IF_RANK_0(system_params::myid, printf("Time to set system coefficients %lf \n", sysmat_end_time));



    /////  check, Fix map and dim
    if (seqStar <= 1 && astro_params::nAstroPSolved > 0)
    {
        printf("ERROR PE=%d Only %d star Run not allowed with this PE numbers .n", system_params::myid, seqStar);
        exit(EXIT_FAILURE);
    }
    comlsqr.VrIdAstroPDim = seqStar;
    long tempDimMax = comlsqr.VrIdAstroPDim;
    long tempVrIdAstroPDimMax;
    MPI_Allreduce(&tempDimMax, &tempVrIdAstroPDimMax, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    comlsqr.VrIdAstroPDimMax = tempVrIdAstroPDimMax;

    comlsqr.mapStar = (int **)calloc(system_params::nproc, sizeof(int *));

    find_mapStar(comlsqr.mapStar, system_params::nproc, system_params::myid, firstStar, lastStar);

    if (comlsqr.mapStar[system_params::myid][0] == comlsqr.mapStar[system_params::myid][1] && astro_params::nAstroPSolved > 0)
    {
        printf("PE=%d ERROR. Only one star in this PE: starting star=%d ending start=%d\n", system_params::myid, comlsqr.mapStar[system_params::myid][0], comlsqr.mapStar[system_params::myid][1]);
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(EXIT_FAILURE);
    }

    if (system_params::myid == 0)
        for (int i = 0; i < system_params::nproc; i++)
            printf("mapStar[%d][0]=%d mapStar[%d][1]=%d\n", i, comlsqr.mapStar[i][0], i, comlsqr.mapStar[i][1]);

    ////////  Check Null Space Vector
    if (constr_params::extConstraint)
    {
        seconds[4] = time(NULL);

        nullSpaceCk = cknullSpace(lsqr_arrs::sysmatAstro, lsqr_arrs::sysmatAtt, lsqr_arrs::sysmatInstr, lsqr_arrs::sysmatGloB, lsqr_arrs::sysmatConstr, lsqr_arrs::matrixIndexAstro, lsqr_arrs::matrixIndexAtt, attNS, comlsqr);

        if (system_params::myid == 0)
        {
            printf("NullSpace check\n");
            for (int j = 0; j < constr_params::nEqExtConstr; j++)
                printf("Eq. Constraint %d: Norm=%15.7f Min=%15.7f Max=%15.7f Avg=%15.7f Var=%15.7f\n", j,
                       nullSpaceCk.vectNorm[j], nullSpaceCk.compMin[j], nullSpaceCk.compMax[j], nullSpaceCk.compAvg[j],
                       nullSpaceCk.compVar[j]);
        }
        seconds[5] = time(NULL);
        tot_sec[4] = seconds[5] - seconds[4];
        if (system_params::myid == 0)
            printf("Time to check nullspace: %ld\n", tot_sec[4]);
        delete[] attNS;
    }

    astro_params::VrIdAstroPDimMax = comlsqr.VrIdAstroPDimMax;
    astro_params::VrIdAstroPDim = comlsqr.VrIdAstroPDim;
    att_params::VroffsetAttParam = astro_params::VrIdAstroPDimMax * astro_params::nAstroPSolved;
    comlsqr.VroffsetAttParam = att_params::VroffsetAttParam;

    system_params::nunkSplit = astro_params::VrIdAstroPDimMax * astro_params::nAstroPSolved + att_params::nAttParam + instr_params::nInstrParam + glob_params::nGlobalParam;
    printf("PE = %d, VrIdAstroPDimMax = %ld, VrIdAstroPDim = %ld, nAstroPSolved = %d, nAttParam = %d, nInstrParam = %d, nGlobalParam = %d\n",
           system_params::myid, astro_params::VrIdAstroPDimMax, astro_params::VrIdAstroPDim, astro_params::nAstroPSolved, att_params::nAttParam, instr_params::nInstrParam, glob_params::nGlobalParam);
    comlsqr.nunkSplit = system_params::nunkSplit;
    printf("nunkSplit = %ld\n", system_params::nunkSplit);


    lsqr_arrs::allocate_vVect_vectors(system_params::nunkSplit,system_params::myid);

    system_params::totmem += (6 * system_params::nunkSplit * sizeof(double)) / (1024 * 1024); // dcopy+vAuxVect locally allocated on lsqr.c

    // Compute and write the total memory allocated
    ACTION_IF_RANK_0(system_params::myid, printf("LOCAL %ld MB of memory allocated on each task.\nTOTAL MB memory allocated= %ld\n", system_params::totmem, system_params::nproc * system_params::totmem));

    int localNumberMapStar = (int)(covStarStarCounter + covStarOtherCounter);

    MPI_Reduce(&localNumberMapStar, &numberOfCompCovEle, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    numberOfCompCovEle = numberOfCompCovEle + covOtherOtherCounter;
    MPI_Barrier(MPI_COMM_WORLD);

    // Compute the preconditioning vector for the system columns

    double precondvec_start_time = MPI_Wtime();
    compute_system_preconditioning(mapNoss, lsqr_arrs::matrixIndexAstro, lsqr_arrs::matrixIndexAtt, lsqr_arrs::sysmatAstro, lsqr_arrs::sysmatAtt, lsqr_arrs::sysmatInstr, lsqr_arrs::sysmatGloB, lsqr_arrs::sysmatConstr, comlsqr.mapStar, lsqr_arrs::preCondVect);
    double precondvec_end_time = MPI_Wtime() - precondvec_start_time;

    ACTION_IF_RANK_0(system_params::myid, printf("Time precondvector %21.10lf \n", precondvec_end_time));

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, &lsqr_arrs::preCondVect[astro_params::VrIdAstroPDimMax * astro_params::nAstroPSolved], att_params::nAttParam + instr_params::nInstrParam + glob_params::nGlobalParam, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 0; i < att_params::nAttParam + instr_params::nInstrParam + glob_params::nGlobalParam; i++)
    {
        if (lsqr_arrs::preCondVect[astro_params::VrIdAstroPDimMax * astro_params::nAstroPSolved + i] == 0.0)
        {
            printf("PE=%d preCondVect[%ld]=0!!\n", system_params::myid, astro_params::VrIdAstroPDimMax * astro_params::nAstroPSolved + i);
        }
    }

    if (astro_params::nAstroPSolved)    SumCirc(lsqr_arrs::preCondVect, comlsqr);

    ////////// TEST for NO ZERO Column on A matrix

    testing_well_poseness(lsqr_arrs::preCondVect, lsqr_input::precond, system_params::myid, astro_params::VrIdAstroPDim, astro_params::VrIdAstroPDimMax, att_params::nAttParam, instr_params::nInstrParam, glob_params::nGlobalParam, astro_params::nAstroPSolved);

    ACTION_IF_RANK_0(system_params::myid, printf("Computation of the preconditioning vector finished.\n"));

    comlsqr.offsetInstrParam = instr_params::offsetInstrParam;
    comlsqr.offsetGlobParam = glob_params::offsetGlobParam;
    comlsqr.nAttParam = att_params::nAttParam;

    initThread(&comlsqr);

    endTime = MPI_Wtime() - endTime;
    MPI_Barrier(MPI_COMM_WORLD);

    /////////////// MAIN CALL
    double precont_start_time = MPI_Wtime();
    precondSystemMatrix(lsqr_arrs::sysmatAstro, lsqr_arrs::sysmatAtt, lsqr_arrs::sysmatInstr, lsqr_arrs::sysmatGloB, lsqr_arrs::sysmatConstr, lsqr_arrs::preCondVect, lsqr_arrs::matrixIndexAstro, lsqr_arrs::matrixIndexAtt, lsqr_arrs::instrCol, comlsqr);
    double precond_end_time = MPI_Wtime() - precont_start_time;

    ACTION_IF_RANK_0(system_params::myid, printf("time routing precondSystemMatrix %21.10lf \nSystem built, ready to call LSQR.\nPlease wait. System solution is underway...", precond_end_time));

    startTime = MPI_Wtime();
    seconds[1] = time(NULL);
    tot_sec[0] = seconds[1] - seconds[0];
    comlsqr.totSec = tot_sec[0];
    ACTION_IF_RANK_0(system_params::myid, printf("Starting lsqr after sec: %ld\n", tot_sec[0]));

    lsqr(system_params::nobs, system_params::nunk, lsqr_input::damp, lsqr_arrs::knownTerms, lsqr_arrs::vVect, lsqr_arrs::wVect,
         lsqr_arrs::xSolution, lsqr_arrs::standardError, lsqr_input::atol, lsqr_input::btol,
         lsqr_input::conlim, lsqr_input::itnlim, &(lsqr_output::istop), &(lsqr_output::itn), &(lsqr_output::anorm),
         &(lsqr_output::acond), &(lsqr_output::rnorm), &(lsqr_output::arnorm),
         &(lsqr_output::xnorm), lsqr_arrs::sysmatAstro, lsqr_arrs::sysmatAtt, lsqr_arrs::sysmatInstr, lsqr_arrs::sysmatGloB, lsqr_arrs::sysmatConstr, lsqr_arrs::matrixIndexAstro,
         lsqr_arrs::matrixIndexAtt, lsqr_arrs::instrCol, lsqr_arrs::instrConstrIlung, comlsqr);

    MPI_Barrier(MPI_COMM_WORLD);
    endTime = MPI_Wtime();
    double clockTime = MPI_Wtime();
    ACTION_IF_RANK_0(system_params::myid, printf("Global Time for lsqr =%f sec \n", endTime - startTime));

    seconds[2] = time(NULL);

   
    auto res=de_preconditioning(lsqr_arrs::xSolution,lsqr_arrs::standardError,lsqr_arrs::preCondVect,system_params::nunkSplit,att_params::VroffsetAttParam,astro_params::VrIdAstroPDim,system_params::myid,lsqr_input::idtest,astro_params::nAstroPSolved);


    seconds[3] = time(NULL);

    if (system_params::myid == 0 && lsqr_input::idtest)
    {
        double average = std::get<0>(res) / system_params::nunk;
        printf("Average deviation from ID solution: %le.\n                               rms: %le.\nMaximum deviation from ID solution: %le.\n",
               average, pow(std::get<1>(res) / system_params::nunk - pow(average, 2.0), 0.5), std::get<2>(res));
    }

    if (system_params::myid == 0)
    {
        tot_sec[1] = seconds[2] - seconds[1];
        tot_sec[2] = seconds[3] - seconds[2];
        tot_sec[3] = seconds[3] - seconds[0];
        printf("Processing finished.\n");
        for (long int ii = 0; ii < 10; ii++)
            printf("xSolution[%ld]=%16.12le \t standardError[%ld]=%16.12le\n", ii, lsqr_arrs::xSolution[ii], ii, lsqr_arrs::standardError[ii]);

        printf("\nEnd run.\ntime before lsqr in sec: %ld\ntime for lsqr: %ld\ntime after lsqr in sec: %ld\ntime TOTAL: %ld\n",
               tot_sec[0], tot_sec[1], tot_sec[2], tot_sec[3]);
    }

    lsqr_arrs::free_system_matrix();
    lsqr_arrs::free_matrix_topology();
    lsqr_arrs::free_memory();

    MPI_Finalize();
    exit(EXIT_SUCCESS);
}