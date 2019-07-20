
#======================================================================

import TasmanianSG
import numpy as np
from parameters import *
import nonlinear_solver_iterate as solveriter

#======================================================================

def ad_grid_iter(n_agents, iDepth, valold):
    # valold is a list

    grid  = TasmanianSG.TasmanianSparseGrid()

    k_range=np.array([k_bar, k_up])

    ranges=np.empty((n_agents, 2))

    for i in range(n_agents):
        ranges[i]=k_range

    iDim=n_agents
    iOut=1
    # TODO: parameterize this
    refinement_level = 1
    fTol = 1.E-5

    # level of grid before refinement
    grid.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")
    grid.setDomainTransform(ranges)

    aPoints=grid.getPoints()
    iNumP1=aPoints.shape[0]
    aVals=np.empty([iNumP1, 1])
    aVals1=np.empty([iNumP1, 1])

    for iI in range(iNumP1):
        aValTemp = 0
        for sIdx in range(5):
            aValTemp += (1./5) * solveriter.iterate(aPoints[iI], n_agents, valold[sIdx], sIdx)[0]
        aVals[iI]=aValTemp
    #print(aVals)
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
                aValTemp += (1./5) * solveriter.iterate(aPoints[iI], n_agents, valold[sIdx], sIdx)[0]
            aVals[iI]=aValTemp
            v=aVals[iI]*np.ones((1,1))
            to_print=np.hstack((aPoints[iI].reshape(1,n_agents), v))
            np.savetxt(file, to_print, fmt='%2.16f')

        print(aVals)
        file.close()
        grid.loadNeededPoints(aVals)

    f=open("grid_iter.txt", 'w')
    np.savetxt(f, aPoints, fmt='% 2.16f')
    f.close()

    return grid

#======================================================================
