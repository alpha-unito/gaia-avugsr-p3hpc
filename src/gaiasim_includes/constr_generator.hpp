#include <mpi.h>
#include "namespaces.hpp"

void generate_external_constraints(double *const &sysmatConstr, 
                                   double *const &knownTerms, 
                                   const long *const &mapNoss, 
                                   double *&attNS, 
                                   int addElementextStar, 
                                   int addElementAtt)
{
    double randVal;
    double *accumulator;

    accumulator = fast_allocate_vector<double>(constr_params::nEqExtConstr);
    attNS = allocate_vector<double>(att_params::nDegFreedomAtt, "attNS", system_params::myid);

    if (system_params::myid == 0)
    {
        for (int i = 0; i < att_params::nDegFreedomAtt; i++)
            attNS[i] = (((double)rand()) / RAND_MAX) * 2 - 1.0;
    }
    MPI_Bcast(attNS, att_params::nDegFreedomAtt, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int j = 0; j < constr_params::nEqExtConstr; j++)
    {
        for (int i = 0; i < addElementextStar + addElementAtt; i++)
        {
            randVal = (((double)rand()) / RAND_MAX) * 2 - 1.0;
            if (i < addElementextStar)
            {
                if (astro_params::nAstroPSolved == 3 && i % astro_params::nAstroPSolved == 0)
                    randVal = 0.;
                if (astro_params::nAstroPSolved == 4 && i % astro_params::nAstroPSolved >= 2)
                    randVal = 0.;
                if (astro_params::nAstroPSolved == 5 && i % astro_params::nAstroPSolved == 0)
                    randVal = 0.;
                if (astro_params::nAstroPSolved == 5 && i % astro_params::nAstroPSolved > 2 && j < 3)
                    randVal = 0.;
            }
            if (i >= addElementextStar)
            {
                if (j < 3)
                    randVal = 1.0;
                if (j == 0 || j == 3)
                {
                    if (i >= addElementextStar + addElementAtt / att_params::nAttAxes)
                        randVal = 0.0;
                }
                if (j == 1 || j == 4)
                {
                    if (i < addElementextStar + addElementAtt / att_params::nAttAxes)
                        randVal = 0.0;
                    if (i >= addElementextStar + 2 * addElementAtt / att_params::nAttAxes)
                        randVal = 0.0;
                }
                if (j == 2 || j == 5)
                {
                    if (i < addElementextStar + 2 * addElementAtt / att_params::nAttAxes)
                        randVal = 0.0;
                }
            }
            sysmatConstr[j * constr_params::nOfElextObs + i] = randVal * att_params::extConstrW;
            accumulator[j] += randVal * att_params::extConstrW;
        }
        if (!lsqr_input::idtest)
            lsqr_arrs::knownTerms[mapNoss[system_params::myid] + j] = 0.;
    } // j=0
    if (lsqr_input::idtest)
        MPI_Allreduce(accumulator, &lsqr_arrs::knownTerms[mapNoss[system_params::myid]], constr_params::nEqExtConstr, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[] accumulator;
}

void generate_bar_constraints(double *const &sysmatConstr, 
                              double *const &knownTerms, 
                              const long *const &mapNoss, 
                              int addElementbarStar)
{
    double randVal;

    double *accumulator=fast_allocate_vector<double>(constr_params::nEqBarConstr);

    for (int j = 0; j < constr_params::nEqBarConstr; j++)
    {
        for (int i = 0; i < addElementbarStar; i++)
        {
            randVal = (((double)rand()) / RAND_MAX) * 2 - 1.0;
            if (astro_params::nAstroPSolved == 3 && i % astro_params::nAstroPSolved == 0)
                randVal = 0.;
            if (astro_params::nAstroPSolved == 4 && i % astro_params::nAstroPSolved >= 2)
                randVal = 0.;
            if (astro_params::nAstroPSolved == 5 && i % astro_params::nAstroPSolved == 0)
                randVal = 0.;
            if (astro_params::nAstroPSolved == 5 && i % astro_params::nAstroPSolved > 2 && j < 3)
                randVal = 0.;

            sysmatConstr[constr_params::nEqExtConstr * constr_params::nOfElextObs + j * constr_params::nOfElBarObs + i] = randVal * att_params::barConstrW;
            accumulator[j] += randVal * att_params::barConstrW;
        }
        if (!lsqr_input::idtest)
            lsqr_arrs::knownTerms[mapNoss[system_params::myid] + constr_params::nEqExtConstr + j] = 0.;
    } // j=0
    if (lsqr_input::idtest)
        MPI_Allreduce(accumulator, &lsqr_arrs::knownTerms[mapNoss[system_params::myid] + constr_params::nEqExtConstr], constr_params::nEqBarConstr, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    delete[] accumulator;
}

void generate_instr_constraints(struct comData comlsqr, 
                                double *const &sysmatConstr, 
                                double *const &knownTerms, 
                                const long *const &mapNoss){
                                    
    double *instrCoeffConstr = fast_allocate_vector<double>(astro_params::nElemIC);
    int *instrColsConstr = fast_allocate_vector<int>(astro_params::nElemIC);

    if (!computeInstrConstr(comlsqr, instrCoeffConstr, instrColsConstr, lsqr_arrs::instrConstrIlung))
    {
        printf("SEVERE ERROR PE=0 computeInstrConstr failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(EXIT_FAILURE);
    }
    //////////////////////////
    for (int k = 0; k < astro_params::nElemIC; k++)
        instrCoeffConstr[k] = instrCoeffConstr[k] * instr_params::wgInstrCoeff;
    /////////////////////////
    for (int j = 0; j < astro_params::nElemIC; j++)
    {
        sysmatConstr[constr_params::nOfElextObs * constr_params::nEqExtConstr + constr_params::nOfElBarObs * constr_params::nEqBarConstr + j] = instrCoeffConstr[j];
        lsqr_arrs::instrCol[mapNoss[system_params::myid] * astro_params::nInstrPSolved + j] = instrColsConstr[j];
    }
    int counter0 = 0;
    for (int j = 0; j < astro_params::nOfInstrConstr; j++)
    {
        double sumVal = 0.;
        for (int k = 0; k < lsqr_arrs::instrConstrIlung[j]; k++)
        {
            sumVal += sysmatConstr[constr_params::nOfElextObs * constr_params::nEqExtConstr + constr_params::nOfElBarObs * constr_params::nEqBarConstr + counter0];
            counter0++;
        }
        if (lsqr_input::idtest)
        {
            lsqr_arrs::knownTerms[mapNoss[system_params::myid] + constr_params::nEqExtConstr + constr_params::nEqBarConstr + j] = sumVal;
        }
        else
        {
            lsqr_arrs::knownTerms[mapNoss[system_params::myid] + constr_params::nEqExtConstr + constr_params::nEqBarConstr + j] = 0.;
        }
    }
    if (counter0 != astro_params::nElemIC)
    {
        printf("SEVERE ERROR PE=0 counter0=%d != nElemIC=%d when computing knownTerms for InstrConstr\n", counter0,
               astro_params::nElemIC);
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(EXIT_FAILURE);
    }

    delete[] instrCoeffConstr;
    delete[] instrColsConstr;
}