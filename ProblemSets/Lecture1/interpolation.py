

import TasmanianSG
import numpy as np
from parameters import *
import nonlinear_solver_initial as solver

#======================================================================

def sparse_grid(n_agents, iDepth, sIdx):

    grid  = TasmanianSG.TasmanianSparseGrid()

    k_range=np.array([k_bar, k_up])

    ranges=np.empty((n_agents, 2))


    for i in range(n_agents):
        ranges[i]=k_range

    iDim=n_agents
    # TODO: parameterize this
    refinement_level = 1
    fTol = 1.E-5

    grid.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")
    grid.setDomainTransform(ranges)

    aPoints=grid.getPoints()
    iNumP1=aPoints.shape[0]
    aVals=np.empty([iNumP1, 1])

    for iI in range(iNumP1):
        aValTemp = 0
        for sIdx in range(5):
            aValTemp += (1./5) * solver.initial(aPoints[iI], n_agents, sIdx)[0]
        aVals[iI]=aValTemp
    grid.loadNeededPoints(aVals)


    for ik in range(refinement_level):
        grid.setSurplusRefinement(fTol, 1, "fds")   #also use fds, or other rules
        aPoints = grid.getNeededPoints()
        iNumP1=aPoints.shape[0]
        aVals=np.empty([iNumP1, 1])

        print(ik)

        file=open("comparison1.txt", 'w')
        for iI in range(iNumP1):
            aValTemp = 0
            for sIdx in range(5):
                aValTemp += (1./5) * solver.initial(aPoints[iI], n_agents, sIdx)[0]
            aVals[iI]=aValTemp
            v=aVals[iI]*np.ones((1,1))
            to_print=np.hstack((aPoints[iI].reshape(1,n_agents), v))
            np.savetxt(file, to_print, fmt='%2.16f')

        print(aVals)
        file.close()
        grid.loadNeededPoints(aVals)

    f=open("grid.txt", 'w')
    np.savetxt(f, aPoints, fmt='% 2.16f')
    f.close()

    return grid
#======================================================================
