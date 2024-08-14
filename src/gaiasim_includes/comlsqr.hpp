#ifndef GAIA2_COMLSQR
#define GAIA2_COMLSQR
#include "namespaces.hpp"

inline void init_comlsqr(struct comData *comlsqr, long int *mapNoss, long int *mapNcoeff, int itnCPR, int itnCPRstop)
{
    comlsqr->nullSpaceAttfact = 1.0 * att_params::attExtConstrFact * att_params::extConstrW;
    comlsqr->barConstrW = att_params::barConstrW;
    comlsqr->extConstrW = att_params::extConstrW;
    comlsqr->nStar = astro_params::nStar;
    comlsqr->nAstroP = astro_params::nAstroP;
    comlsqr->nAstroPSolved = astro_params::nAstroPSolved;
    comlsqr->nAttP = att_params::nAttP;
    comlsqr->nInstrPSolved = astro_params::nInstrPSolved;
    comlsqr->nGlobP = glob_params::nGlobP;
    comlsqr->mapNossBefore = system_params::mapNossBefore;
    comlsqr->mapNossAfter = system_params::mapNossAfter;
    comlsqr->myid = system_params::myid;
    comlsqr->nproc = system_params::nproc;
    comlsqr->mapNoss = mapNoss;
    comlsqr->mapNcoeff = mapNcoeff;
    comlsqr->nAttParam = att_params::nAttParam;
    comlsqr->extConstraint = constr_params::extConstraint;
    comlsqr->nEqExtConstr = constr_params::nEqExtConstr;
    comlsqr->numOfExtStar = constr_params::numOfExtStar;
    comlsqr->barConstraint = constr_params::barConstraint;
    comlsqr->nEqBarConstr = constr_params::nEqBarConstr;
    comlsqr->numOfBarStar = constr_params::numOfBarStar;
    comlsqr->firstStarConstr = constr_params::firstStarConstr;
    comlsqr->lastStarConstr = constr_params::lastStarConstr;
    comlsqr->numOfExtAttCol = att_params::numOfExtAttCol;
    comlsqr->startingAttColExtConstr = constr_params::startingAttColExtConstr;
    comlsqr->setBound[0] = 0;
    comlsqr->setBound[1] = astro_params::nAstroPSolved;
    comlsqr->setBound[2] = astro_params::nAstroPSolved + att_params::nAttP;
    comlsqr->setBound[3] = astro_params::nAstroPSolved + att_params::nAttP + astro_params::nInstrPSolved;
    comlsqr->nDegFreedomAtt = att_params::nDegFreedomAtt;
    comlsqr->nAttParAxis = att_params::nAttParAxis;
    comlsqr->nAttAxes = att_params::nAttAxes;
    comlsqr->nobs = system_params::nobs;
    comlsqr->lsInstrFlag = instr_params::lsInstrFlag;
    comlsqr->ssInstrFlag = instr_params::ssInstrFlag;
    comlsqr->nuInstrFlag = instr_params::nuInstrFlag;
    comlsqr->maInstrFlag = instr_params::maInstrFlag;
    comlsqr->cCDLSAACZP = att_params::cCDLSAACZP;
    comlsqr->nOfInstrConstr = astro_params::nOfInstrConstr;
    comlsqr->nElemIC = astro_params::nElemIC;
    comlsqr->nElemICLSAL = astro_params::nElemICLSAL;
    comlsqr->nElemICLSAC = astro_params::nElemICLSAC;
    comlsqr->nElemICSS = astro_params::nElemICSS;
    comlsqr->instrConst[0] = instr_params::instrConst[0];
    comlsqr->instrConst[1] = instr_params::instrConst[1];
    comlsqr->instrConst[2] = instr_params::instrConst[2];
    comlsqr->instrConst[3] = instr_params::instrConst[3];
    comlsqr->nInstrParam = instr_params::nInstrParam;
    comlsqr->nGlobalParam = glob_params::nGlobalParam;
    comlsqr->timeCPR = DEFAULT_TIMECPR;
    comlsqr->timeLimit = DEFAULT_TIMELIMIT;
    comlsqr->itnCPR = itnCPR;
    comlsqr->itnCPRstop = itnCPRstop;
    comlsqr->itnLimit = lsqr_input::itnlim;
    comlsqr->Test = 0; // it is not a running test but a production run
    comlsqr->instrConst[0] = instr_params::instrConst[0];
    comlsqr->instrConst[1] = instr_params::instrConst[1];
    comlsqr->instrConst[2] = instr_params::instrConst[2];
    comlsqr->instrConst[3] = instr_params::instrConst[3];
    comlsqr->nvinc = 0;
    comlsqr->parOss = (long)system_params::nparam;
    comlsqr->nunk = system_params::nunk;
    comlsqr->offsetAttParam = att_params::offsetAttParam;
}

inline void set_comlsqr_offsets(struct comData *comlsqr, int nCCDs, int nFoVs, int nPixelColumns, int nTimeIntervals){
    comlsqr->offsetCMag = instr_params::maInstrFlag * nCCDs;                                           // offest=0 if maInstrFlag=0
    comlsqr->offsetCnu = comlsqr->offsetCMag + instr_params::nuInstrFlag * nFoVs * nCCDs;               // offest=offsetCMag if nuInstrFlag=0
    comlsqr->offsetCdelta_eta = comlsqr->offsetCnu + instr_params::ssInstrFlag * nCCDs * nPixelColumns; // offest=offsetCnu if ssInstrFlag=0
    comlsqr->offsetCDelta_eta_1 = comlsqr->offsetCdelta_eta + instr_params::lsInstrFlag * nFoVs * nCCDs * nTimeIntervals;
    comlsqr->offsetCDelta_eta_2 = comlsqr->offsetCdelta_eta + instr_params::lsInstrFlag * 2 * nFoVs * nCCDs * nTimeIntervals;
    comlsqr->offsetCDelta_eta_3 = comlsqr->offsetCdelta_eta + instr_params::lsInstrFlag * 3 * nFoVs * nCCDs * nTimeIntervals;
    comlsqr->offsetCdelta_zeta = comlsqr->offsetCDelta_eta_3 + instr_params::ssInstrFlag * nCCDs * nPixelColumns;
    comlsqr->offsetCDelta_zeta_1 = comlsqr->offsetCdelta_zeta + instr_params::lsInstrFlag * nFoVs * nCCDs * nTimeIntervals;
    comlsqr->offsetCDelta_zeta_2 = comlsqr->offsetCdelta_zeta + instr_params::lsInstrFlag * 2 * nFoVs * nCCDs * nTimeIntervals;
    comlsqr->nInstrPSolved = astro_params::nInstrPSolved;
}

#endif