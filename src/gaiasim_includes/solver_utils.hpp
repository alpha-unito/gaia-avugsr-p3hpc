#ifndef GAIA2_SOLVER_UTILS
#define GAIA2_SOLVER_UTILS

#include <iostream>

#include "namespaces.hpp"

#define ACTION_IF_RANK_0_ELSE(rank, action, elseaction) \
    (rank == 0) ? action : elseaction;

#define ACTION_IF_RANK_0(rank, action) \
    if (rank == 0)                     \
        action;

inline void parse_cli(int argc, char **argv, int *zeroAtt, int *zeroInstr, int *zeroGlob)
{
    if (argc == 1)
    {
        ACTION_IF_RANK_0(system_params::myid, printf("Usage:  solvergaiaSim  [-memGlobal value]  [-IDtest [value]   -noFile -noConstr -numFilexproc nfileProc -Precond [on|off] -timeCPR hours -timelimit hours -itnCPR numberOfIterations -nth numoerOfOMPThreads -itnlimit numberOfIterations   -atol value  -inputDir inputdir -outputDir outputdir -zeroAtt -zeroInstr -zeroGlob -wgic value]] -wrfilebin writedir  -extConstr weight -barConstr weight \n\n"));
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "-memGlobal") == 0)
        {
            i++;
            system_params::memGlobal = atof(argv[i]);
            printf("-memGlobal %f", system_params::memGlobal);
        }
        else if (strcmp(argv[i], "-Precond") == 0)
        {
            i++;
            lsqr_input::precond = (strcmp(argv[i], "off") != 0);
        }
        else if (strcmp(argv[i], "-itnlimit") == 0)
        {
            int testTime = 0;
            i++;
            testTime = atoi(argv[i]);
            if (testTime > 0)
            {
                system_params::itnLimit = testTime;
                lsqr_input::itnlim = testTime;
            }
        }
        else if (strcmp(argv[i], "-atol") == 0)
        {
            i++;
            double dummy = atof(argv[i]);
            if (dummy >= 0)
                lsqr_input::aTol = dummy;
        }
        else if (strcmp(argv[i], "-zeroAtt") == 0)
        {
            *zeroAtt = 1;
        }
        else if (strcmp(argv[i], "-zeroInstr") == 0)
        {
            *zeroInstr = 1;
        }
        else if (strcmp(argv[i], "-zeroGlob") == 0)
        {
            *zeroGlob = 1;
        }
        else if (strcmp(argv[i], "-wgic") == 0)
        {
            i++;
            int tmp;
            instr_params::wgInstrCoeff = (tmp = atoi(argv[i]) > 0) ? tmp : 1;
        }
        else if (strcmp(argv[i], "-extConstr") == 0)
        {
            i++;
            constr_params::extConstraint = 1; // extendet external constraint
            constr_params::noConstr = 2;
            constr_params::nEqExtConstr = DEFAULT_EXTCONSTROWS;
            att_params::extConstrW = atof(argv[i]);
            if (att_params::extConstrW == 0.0)
            {
                printf("ERROR: PE=%d -extConstr option given with no value or zero value ==> %le. Aborting\n", system_params::myid, att_params::extConstrW);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-barConstr") == 0)
        {
            i++;
            constr_params::barConstraint = 1; // extendet external constraint
            constr_params::noConstr = 2;
            constr_params::nEqBarConstr = DEFAULT_BARCONSTROWS;
            att_params::barConstrW = atof(argv[i]);
            if (att_params::barConstrW == 0.0)
            {
                printf("ERROR: PE=%d -barConstr option given with no value or zero value ==> %le. Aborting\n", system_params::myid, att_params::barConstrW);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (constr_params::extConstraint && constr_params::barConstraint)
    {
        printf("Error: baricentric anld null space constraints are mutually exclusive. Aborting\n");
        exit(EXIT_FAILURE);
    }
}

inline void fill_system(double *const &sysmatAstro,
                        double *const &sysmatAtt,
                        double *const &sysmatInstr,
                        double *const &sysmatGloB,
                        const long *const &mapNoss,
                        long *const &matrixIndexAstro,
                        long *const &matrixIndexAtt,
                        const int zeroAtt,
                        const int zeroInstr)
{

    long nObsxStar = system_params::nobs / astro_params::nStar;
    long nobsOri = system_params::nobs;
    long startingStar = 0;
    long obsTotal = 0;
    long startFreedom = (att_params::nDegFreedomAtt / system_params::nproc) * system_params::myid;
    long endFreedom = startFreedom + (att_params::nDegFreedomAtt / system_params::nproc) + 1;
    long lastFreedom = startFreedom;
    long instrStartFreedom = (instr_params::nInstrParam / system_params::nproc) * system_params::myid;
    long instrEndFreedom = instrStartFreedom + (instr_params::nInstrParam / system_params::nproc) + 1;
    if (system_params::myid == system_params::nproc - 1)
        instrEndFreedom = instr_params::nInstrParam - 1;
    long currentStar = 0;

    int constraintFound[MAX_CONSTR][2]; // first: number of file of the constraint; second: row in file
    int residual = 0;
    int obsStarnow;
    int numOfObslast = 0;
    int freedomReached = 0;
    int instrFreedomReached = 0;
    int isConstraint = 0;
    int instrLastFreedom = instrStartFreedom;
    int obsStar = 0;

    /////////
    for (long p = 0; p < system_params::myid; p++)
    {
        while (obsTotal < mapNoss[p])
        {
            if (residual == 0)
            {
                obsTotal += nObsxStar;
                if (startingStar < nobsOri % astro_params::nStar)
                    obsTotal++;
            }
            else
            { 
                obsTotal = residual;
                residual = 0;
            }

            if (obsTotal <= mapNoss[p])
                startingStar++;
        } 
        residual = obsTotal - mapNoss[p];
        obsTotal = 0;
    } 
    //////////////////////////

    //////////////// filling the system
    currentStar = startingStar;
    obsStar = residual;

    srand(system_params::myid);

    if (obsStar == 0)
    {
        obsStar = nObsxStar;
        if (currentStar < nobsOri % astro_params::nStar)
            obsStar++;
    }
    obsStarnow = obsStar;

    int offsetConstraint = isConstraint - obsStar; // number of constraint alredy computed in the previous PE
    if (offsetConstraint < 0)
        offsetConstraint = 0;

    int counterStarObs = 0;
    int rowInFile = -1;
    int changedStar = 0;
    int counterConstr = 0;
    //////////////////////////////////////////////////////////////////
    ///////////   RUNNING ON ALL OBSERVATIONS
    //////////////////////////////////////////////////////////////////
    for (long ii = 0, ic = 0; ii < mapNoss[system_params::myid]; ii++)
    {
        rowInFile++;
        if (currentStar % 1000 == 0 && changedStar)
        {
            rowInFile = 0;
            changedStar = 0;
        }
        ///////////// generate MatrixIndex
        if (currentStar == astro_params::nStar)
        {
            printf("PE=%d Severe Error in currentStar=%ld ii=%ld\n", system_params::myid, currentStar, ii);
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(EXIT_FAILURE);
        }

        if (astro_params::nAstroPSolved)
            matrixIndexAstro[ii] = currentStar * astro_params::nAstroPSolved;

        if (!freedomReached && astro_params::nAstroPSolved)
        {
            if ((obsStar - counterStarObs) <= isConstraint)
            { // constraint
                matrixIndexAtt[ii] = att_params::offsetAttParam;
                constraintFound[counterConstr][0] = currentStar / 1000;
                constraintFound[counterConstr][1] = rowInFile;
                counterConstr++;

                if (counterConstr == MAX_CONSTR)
                {
                    printf("PE=%d Abort increase MAX_CONSTR and recompile =%d \n", system_params::myid, counterConstr);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    exit(EXIT_FAILURE);
                }
            }
            else
            {
                if (lastFreedom >= att_params::nDegFreedomAtt - att_params::nAttParAxis)
                    lastFreedom = att_params::nDegFreedomAtt - att_params::nAttParAxis;
                matrixIndexAtt[ii] = att_params::offsetAttParam + lastFreedom;
                if (lastFreedom >= endFreedom || lastFreedom >= att_params::nDegFreedomAtt - att_params::nAttParAxis)
                    freedomReached = 1;
                lastFreedom += att_params::nAttParAxis;
            }
        }
        else
        {
            lastFreedom = (((double)rand()) / (((double)RAND_MAX))) *
                          (att_params::nDegFreedomAtt - att_params::nAttParAxis + 1);
            if (lastFreedom > att_params::nDegFreedomAtt - att_params::nAttParAxis)
                lastFreedom = att_params::nDegFreedomAtt - att_params::nAttParAxis;
            if ((obsStar - counterStarObs) <= isConstraint) // constraint
            {
                lastFreedom = 0;
                constraintFound[counterConstr][0] = currentStar / 1000;
                constraintFound[counterConstr][1] = rowInFile;
                counterConstr++;
            }
            matrixIndexAtt[ii] = att_params::offsetAttParam + lastFreedom;
        }
        ///////////// generate InstrIndex

        if (!instrFreedomReached && astro_params::nInstrPSolved)
        {
            if ((obsStar - counterStarObs) <= isConstraint)
            { // constraint
                for (int kk = 0; kk < astro_params::nInstrPSolved; kk++)
                    lsqr_arrs::instrCol[ii * astro_params::nInstrPSolved + kk] = 0;
            }
            else
            {
                if (instrLastFreedom > instrEndFreedom)
                    instrLastFreedom = instrEndFreedom;
                lsqr_arrs::instrCol[ii * astro_params::nInstrPSolved] = instrLastFreedom;
                for (int kk = 1; kk < astro_params::nInstrPSolved; kk++)
                    lsqr_arrs::instrCol[ii * astro_params::nInstrPSolved + kk] =
                        (((double)rand()) / (((double)RAND_MAX))) * (instr_params::nInstrParam - 1);
                if (instrLastFreedom == instrEndFreedom)
                    instrFreedomReached = 1;
                instrLastFreedom++;
            }
        }
        else
        {
            if ((obsStar - counterStarObs) <= isConstraint)
            { // constraint
                for (int kk = 0; kk < astro_params::nInstrPSolved; kk++)
                {
                    lsqr_arrs::instrCol[ii * astro_params::nInstrPSolved + kk] = 0;
                }
            }
            else
            {
                for (int kk = 0; kk < astro_params::nInstrPSolved; kk++)
                    lsqr_arrs::instrCol[ii * astro_params::nInstrPSolved + kk] =
                        (((double)rand()) / (((double)RAND_MAX))) * (instr_params::nInstrParam - 1);
            }
        }

        ///////////// generate systemMatrix
        if ((obsStar - counterStarObs) > isConstraint)
        {

            for (short q = 0; q < astro_params::nAstroPSolved; q++)
                sysmatAstro[ii * astro_params::nAstroPSolved + q] = (((double)rand()) / RAND_MAX) * 2 - 1.0;
            for (short q = 0; q < att_params::nAttP; q++)
                sysmatAtt[ii * att_params::nAttP + q] = (((double)rand()) / RAND_MAX) * 2 - 1.0;
            for (short q = 0; q < astro_params::nInstrPSolved; q++)
                sysmatInstr[ii * astro_params::nInstrPSolved + q] = (((double)rand()) / RAND_MAX) * 2 - 1.0;
            for (short q = 0; q < glob_params::nGlobP; q++)
                sysmatGloB[ii * glob_params::nGlobP + q] = (((double)rand()) / RAND_MAX) * 2 - 1.0;
        }
        else 
        {

            for (short q = 0; q < astro_params::nAstroPSolved; q++)
                sysmatAstro[ii * astro_params::nAstroPSolved + q] = 0.0;
            for (short q = 0; q < att_params::nAttP; q++)
                sysmatAtt[ii * att_params::nAttP + q] = 0.0;
            for (short q = 0; q < astro_params::nInstrPSolved; q++)
                sysmatInstr[ii * astro_params::nInstrPSolved + q] = 0.0;
            for (short q = 0; q < glob_params::nGlobP; q++)
                sysmatGloB[ii * glob_params::nGlobP + q] = 0.0;

            if (astro_params::nAstroPSolved > 0)
            {

                if (ii != 0)
                    offsetConstraint = 0;
                int foundedConstraint = (obsStar - counterStarObs) + offsetConstraint;
                int itis = 0;
            }
            ++ic;
        }

        /////////////////////////
        if ((obsStar - counterStarObs) <= isConstraint)
        {
            printf("PE=%d isConstraint=%d ii=%ld matrixIndex[ii*2]=%ld matrixIndex[ii*2+1]=%ld\n", system_params::myid, isConstraint,
                   ii, matrixIndexAstro[ii], matrixIndexAtt[ii]);
            printf("\n");
        }

        /////////////////// Prepare next Obs
        counterStarObs++;
        if (counterStarObs == obsStar)
        {
            if (system_params::myid == (system_params::nproc - 1))
                numOfObslast = counterStarObs;
            counterStarObs = 0;
            currentStar++;
            changedStar = 1;
            isConstraint = 0;
            obsStar = nObsxStar;
            if (currentStar < nobsOri % astro_params::nStar)
                obsStar++;
        }
        ///////////////////////////////// Filling knownTerms  -1.. 1
        if (!lsqr_input::idtest)
            lsqr_arrs::knownTerms[ii] =
                (((double)rand()) / RAND_MAX) * 2.0 - 1.0; // show idtest=1  at the beginning instead of =0
        /////////////////////////////////////////

        ///////////////////////////////////
    } 

    if (!freedomReached && !zeroAtt)
    {
        printf("PE=%d Error ndegFreedomAtt not correctly generated\n", system_params::myid);
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(EXIT_FAILURE);
    }
    if (!instrFreedomReached && !zeroInstr)
    {
        printf("PE=%d Error instrP not all generated instrLastFreedom=%d\n", system_params::myid, instrLastFreedom);
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(EXIT_FAILURE);
    }
}




template <typename T>
inline void work_distribution(const T &mapNoss, const T &mapNcoeff, const long& nobs, const int& nproc, const int& nparam, const int& myid)
{

    long int mapNcoeffBefore{0};
    long int mapNcoeffAfter{0};

    for (int i = 0; i < nproc; i++)
    {
        mapNoss[i] = (nobs)/nproc;
        if (nobs % nproc >= i + 1)
            mapNoss[i]++;
        mapNcoeff[i] = mapNoss[i] * nparam;
        if (i < myid)
        {
            system_params::mapNossBefore += mapNoss[i];
            mapNcoeffBefore += mapNcoeff[i];
        }
        if (i > system_params::myid)
        {
            system_params::mapNossAfter += mapNoss[i];
            mapNcoeffAfter += mapNcoeff[i];
        }
    }
}

void extNobxStarfile(const long *const &mapNoss)
{
    int *sumNObsxStar;
    sumNObsxStar = fast_allocate_vector<int>(astro_params::nStar);
    int irest = system_params::nobs % astro_params::nStar;
    for (int i = 0; i < astro_params::nStar; i++)
    {
        sumNObsxStar[i] = system_params::nobs / astro_params::nStar;
        if (i < irest)
            sumNObsxStar[i]++;
    }

    long counterObsxStar = 0;
    for (int i = 0; i < astro_params::nStar; i++)
    {
        counterObsxStar += sumNObsxStar[i];
        if (counterObsxStar > system_params::mapNossBefore && constr_params::firstStarConstr == -1)
            constr_params::firstStarConstr = i; // first star assigned in  extConstr
        if (counterObsxStar >= system_params::mapNossBefore + mapNoss[system_params::myid] && constr_params::lastStarConstr == -1)
        {
            constr_params::lastStarConstr = i; // last star assigned in  extConstr (it will be eqaul to lastrStar-1 in case of overlap)
            if (counterObsxStar > (system_params::mapNossBefore + mapNoss[system_params::myid]) && system_params::myid != (system_params::nproc - 1))
            {
                constr_params::starOverlap = 1;
                constr_params::lastStarConstr--;
            }
            break;
        }
    }
    constr_params::numOfExtStar = constr_params::lastStarConstr - constr_params::firstStarConstr + 1; // number of stars computed in ext Constr

    int attRes = att_params::nDegFreedomAtt % system_params::nproc;
    constr_params::startingAttColExtConstr = (att_params::nDegFreedomAtt / system_params::nproc) * system_params::myid;
    if (system_params::myid < attRes)
        constr_params::startingAttColExtConstr += system_params::myid;
    else
        constr_params::startingAttColExtConstr += attRes;
    constr_params::endingAttColExtConstr = constr_params::startingAttColExtConstr + (att_params::nDegFreedomAtt / system_params::nproc) - 1;
    if (system_params::myid < attRes)
        constr_params::endingAttColExtConstr++;

    att_params::numOfExtAttCol = constr_params::endingAttColExtConstr - constr_params::startingAttColExtConstr + 1; // numeroi di colonne x asse

    delete[] sumNObsxStar;
}

void barNobxStarfile(const long *const &mapNoss)
{
    int *sumNObsxStar = fast_allocate_vector<int>(astro_params::nStar);
    int irest = system_params::nobs % astro_params::nStar;
    for (int i = 0; i < astro_params::nStar; i++)
    {
        sumNObsxStar[i] = system_params::nobs / astro_params::nStar;
        if (i < irest)
            sumNObsxStar[i]++;
    }

    long counterObsxStar = 0;
    for (int i = 0; i < astro_params::nStar; i++)
    {
        counterObsxStar += sumNObsxStar[i];
        if (counterObsxStar > system_params::mapNossBefore && constr_params::firstStarConstr == -1)
            constr_params::firstStarConstr = i; // first star assigned in  barConstr
        if (counterObsxStar >= system_params::mapNossBefore + mapNoss[system_params::myid] && constr_params::lastStarConstr == -1)
        {
            constr_params::lastStarConstr = i; // last star assigned in  barConstr (it will be eqaul to lastrStar-1 in case of overlap)
            if (counterObsxStar > (system_params::mapNossBefore + mapNoss[system_params::myid]) && system_params::myid != (system_params::nproc - 1))
            {
                constr_params::starOverlap = 1;
                constr_params::lastStarConstr--;
            }
            break;
        }
    }
    constr_params::numOfBarStar = constr_params::lastStarConstr - constr_params::firstStarConstr + 1; // number of stars computed in bar Constr
    delete[] sumNObsxStar;
}

inline void print_params(long int numberOfCovEle)
{
    if (system_params::memGlobal != 0)
    {
        auto memGB = simfullram(&(astro_params::nStar), &system_params::nobs, system_params::memGlobal, system_params::nparam, att_params::nAttParam, instr_params::nInstrParam);
        printf("Running with memory %f GB, nStar=%ld nobs=%ld\n", memGB, astro_params::nStar, system_params::nobs);
    }
    printf("atol= %18.15lf\n", lsqr_input::atol);
    printf("btol= %18.15lf\n", lsqr_input::btol);
    printf("conlim= %18.15le\n", lsqr_input::conlim);
    printf("itnlim= %7ld\n", lsqr_input::itnlim);
    printf("damp= %18.15lf\n", lsqr_input::damp);
    printf("nStar= %7ld\n", astro_params::nStar);
    printf("nAstroP= %7hd\n", astro_params::nAstroP);
    printf("nAstroPSolved= %hd\n", astro_params::nAstroPSolved);
    printf("nDegFreedomAtt= %7ld\n", att_params::nDegFreedomAtt);
    printf("nAttAxes= %7hd\n", att_params::nAttAxes);
    printf("nFovs= %7d ", instr_params::instrConst[0] + 1);
    printf("(instrConst[0])= %7d\n", instr_params::instrConst[0]);
    printf("nCCDs= %7d\n", instr_params::instrConst[1]);
    printf("nPixelColumns= %7d\n", instr_params::instrConst[2]);
    printf("nTimeIntervals= %7d\n", instr_params::instrConst[3]);
    printf("nInstrPSolved= %7hd\n", astro_params::nInstrPSolved);
    printf("lsInstrFlag= %7d\n", instr_params::lsInstrFlag);
    printf("ssInstrFlag= %7d\n", instr_params::ssInstrFlag);
    printf("nuInstrFlag= %7d\n", instr_params::nuInstrFlag);
    printf("maInstrFlag= %7d\n", instr_params::maInstrFlag);
    printf("nOfInstrConstr= %7d\n", astro_params::nOfInstrConstr);
    printf("nElemIC= %7d\n", astro_params::nElemIC);
    printf("nGlobP= %7hd\n", glob_params::nGlobP);
    printf("nObs= %10ld\n", system_params::nobs);
    printf("nCovEle= %ld\n", numberOfCovEle);
}

inline void print_cmdline_input(int nfileProc, int itnCPR, int zeroAtt, int zeroInstr, int zeroGlob, int numberOfCovEle)
{
    printf("Execution of solvergaia Simulator version %s on %d mpi-tasks\n solvergaiaSim ", VER, system_params::nproc);
    if (lsqr_input::idtest)
        printf("-IDtest %le ", lsqr_input::srIDtest);
    if (lsqr_input::precond)
        printf("-Precond on");
    else
        printf("-Precond off");
    if (nfileProc != 3)
        printf("-numFilexproc %d ", nfileProc);
    if (system_params::itnLimit > 0)
        printf(" -itnlimit %d ", system_params::itnLimit);
    if (itnCPR != DEFAULT_ITNCPR)
        printf(" -itnCPR %d ", itnCPR);
    if (zeroAtt)
        printf(" -zeroAtt ");
    if (constr_params::noConstr == 1)
        printf(" -noConstr ");
    if (zeroInstr)
        printf(" -zeroInstr ");
    if (zeroGlob)
        printf(" -zeroGlob ");
    if (constr_params::extConstraint)
        printf("-extConstr %le ", att_params::extConstrW);
    if (constr_params::barConstraint)
        printf("-barConstr %le ", att_params::barConstrW);
    if (numberOfCovEle)
        printf("-nCovEle %ld ", numberOfCovEle);
    if (system_params::nth != 1)
        printf("-nth %d ", system_params::nth);
    printf("-wgic %d", instr_params::wgInstrCoeff);
}

void find_mapStar(int **const &mapStar, const int &nproc, const int &myid, const int &firstStar, const int &lastStar)
{
    int **tempStarSend, **tempStarRecv;
    tempStarSend = (int **)calloc(nproc, sizeof(int *));
    for (int i = 0; i < nproc; i++)
        tempStarSend[i] = (int *)calloc(2, sizeof(int));
    tempStarRecv = (int **)calloc(nproc, sizeof(int *));
    for (int i = 0; i < nproc; i++)
        tempStarRecv[i] = (int *)calloc(2, sizeof(int));

    int *testVectSend, *testVectRecv;
    testVectSend = (int *)calloc(2 * nproc, sizeof(int));
    testVectRecv = (int *)calloc(2 * nproc, sizeof(int));
    testVectSend[2 * myid] = firstStar;
    testVectSend[2 * myid + 1] = lastStar;

    MPI_Allreduce(testVectSend, testVectRecv, 2 * nproc, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < nproc; i++)
        mapStar[i] = (int *)calloc(2, sizeof(int));
    for (int i = 0; i < nproc; i++)
    {
        mapStar[i][0] = testVectRecv[2 * i];
        mapStar[i][1] = testVectRecv[2 * i + 1];
    }

    for (int i = 0; i < nproc; i++)
    {
        free(tempStarSend[i]);
        free(tempStarRecv[i]);
    }

    free(tempStarSend);
    free(tempStarRecv);
}

#endif