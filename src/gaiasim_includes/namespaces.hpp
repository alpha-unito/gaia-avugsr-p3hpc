#ifndef GAIA2_SIM_NAMESPACES
#define GAIA2_SIM_NAMESPACES

#include "allocators.hpp"

// LSQR input parameters
namespace lsqr_input
{
    int itnlim = 2000,
        idtest=1,
        precond=1;
    double damp = 0.0,
           atol = 0.0,
           btol = 0.0,
           conlim = 10000000000000.000000,
           aTol = -1.0,
           srIDtest=0.0;
};

// LSQR output parameters
namespace lsqr_output
{
    int istop;
    int itn;
    double anorm, acond, rnorm, arnorm, xnorm;
};

// LSQR Arrays
namespace lsqr_arrs
{

    /*
        sysmatAstro     -> Astrometric
        sysmatAtt       -> Asset
        sysmatInstr     -> Instrumental
        sysmatGloB      -> Global
        sysmatConstr    -> Constraints
    */
    double *sysmatAstro{nullptr},
            *sysmatAtt{nullptr},
            *sysmatInstr{nullptr},
            *sysmatGloB{nullptr},
            *sysmatConstr{nullptr},
            *knownTerms{nullptr},
            *vVect{nullptr},
            *wVect{nullptr},
            *xSolution{nullptr},
            *standardError{nullptr},
            *preCondVect{nullptr};

    int *instrCol{nullptr},
        *instrConstrIlung{nullptr};
    long *matrixIndexAstro{nullptr},
         *matrixIndexAtt{nullptr};


inline void allocate_system_matrix_knownTerms(const long& mapNoss,const long& nElements,const short& nAstroPSolved, const short& nAttP, const short& nInstrPSolved, const short& nGlobP, const int& nTotConstr, const int& myid){
        allocate_vector<double>(sysmatAstro,mapNoss*nAstroPSolved,"sysmatAstro",myid);
        allocate_vector<double>(sysmatAtt,mapNoss*nAttP,"sysmatAtt",myid);
        allocate_vector<double>(sysmatInstr,mapNoss*nInstrPSolved,"sysmatInstr",myid);
        allocate_vector<double>(sysmatGloB,mapNoss*nGlobP,"sysmatGloB",myid);
        allocate_vector<double>(sysmatConstr,nTotConstr,"sysmatGloB",myid);
        allocate_vector<double>(knownTerms,nElements,"knownTerms",myid);
    }

    inline void allocate_vVect_vectors(const long& nunkSplit, const int& myid){
        allocate_vector<double>(preCondVect,nunkSplit, "preCondVect", myid);
        allocate_vector<double>(vVect,nunkSplit, "vVect", myid);
        allocate_vector<double>(wVect,nunkSplit, "wVect", myid);
        allocate_vector<double>(xSolution,nunkSplit, "xSolution",myid);
        allocate_vector<double>(standardError,nunkSplit, "standardError",myid);
    }

    inline void allocate_matrix_topology(const long& nc,const int& ninstr, const long& mi){
        fast_allocate_vector<int>(instrCol,nc);
        fast_allocate_vector<int>(instrConstrIlung,ninstr);
        fast_allocate_vector<long>(matrixIndexAstro,mi);
        fast_allocate_vector<long>(matrixIndexAtt,mi);
    }

    inline void free_system_matrix(){
        free_mem(sysmatAstro);
        free_mem(sysmatAtt);
        free_mem(sysmatInstr);
        free_mem(sysmatGloB);
        free_mem(sysmatConstr);  
        free_mem(knownTerms);
    }

    inline void free_matrix_topology(){
        free_mem(instrCol);
        free_mem(instrConstrIlung);
        free_mem(matrixIndexAstro);
        free_mem(matrixIndexAtt);
    }

    inline void free_memory(){
        free_mem(preCondVect);
        free_mem(vVect);
        free_mem(wVect);
        free_mem(xSolution);
        free_mem(standardError);
    }
};

namespace system_params
{
    unsigned long int totmem=0;

    long nunk=0,
        nunkSplit=0,
        nElements=0,
        nobs=2400000,
        mapNossBefore=0,
        mapNossAfter=0;

    int nparam=0,
        nproc=0,
        myid=0,
        namelen=0,
        nth=1,
        itnLimit=20000;

    float memGlobal=0.0;

    char processor_name[MPI_MAX_PROCESSOR_NAME];


};

// Astrometrics system parametersß
namespace astro_params
{
    long nStar = 10000,
         nAstroParam=0,
         VrIdAstroPDimMax=0,
         VrIdAstroPDim=0;
    short nAstroP = 5,
          nAstroPSolved = 5,
          nInstrPSolved = 6;
    int *mapAstroP; 
    int nElemIC = -1,
        nOfInstrConstr = -1,
        nOfInstrConstrLSAL = 0,
        nElemICLSAL = 0,
        nOfInstrConstrLSAC = 0,
        nElemICLSAC = 0,
        nOfInstrConstrSS = 0,
        nElemICSS = 0;
};

// Attitude parameters:
namespace att_params
{
    const double attExtConstrFact = -0.5;
    double extConstrW=0.0, barConstrW = 10.0;

    long nDegFreedomAtt = 230489,
         cCDLSAACZP = 14,
         nAttParam=0,
         offsetAttParam=0,
         VroffsetAttParam=0;

    int numOfExtAttCol=0;

    short nAttAxes = 3,
          nAttParAxis = 4,
          nAttP = 0;


};

// Instrumental parameters:
namespace instr_params
{
    long nInstrParam=0,
        offsetInstrParam=0;
    int lsInstrFlag = 1,
        ssInstrFlag = 1,
        nuInstrFlag = 1,
        maInstrFlag = 1,
        wgInstrCoeff = 100;
    
    int instrConst[DEFAULT_NINSTRINDEXES]={1,62,1966,22};


};

// Global parameters:
namespace glob_params
{
    long nGlobalParam=0,
         nGlobP=0,
         offsetGlobParam=0;
};

// Constraints parameters:
namespace constr_params
{

    long lastStarConstr = -1,
        firstStarConstr = -1;

    int extConstraint = 0,
        nEqExtConstr = 0,
        barConstraint = 1,
        nEqBarConstr = DEFAULT_BARCONSTROWS,
        startingAttColExtConstr=0,
        endingAttColExtConstr=0,
        starOverlap = 0,
        numOfExtStar=0,
        numOfBarStar=0,
        noConstr=1,
        nOfElextObs=0,
        nOfElBarObs=0;

};

#endif

