

import matplotlib



# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import TasmanianSG
import numpy as np
import math
from random import uniform
import matplotlib.pyplot as plt

grid  = TasmanianSG.TasmanianSparseGrid()

#############################################################################

# Initialize Test Functions and Parameters
c = np.array([.1, .5])
w = np.array([-.1, .5])
d = 2

def oscillatory(x, c, w, d):
    innersum = 0
    for iI in range(d):
        innersum +=  c[iI] * x[iI]
    return np.cos(2 * np.pi * w[0] + innersum)

def gaussian(x, c, w, d):
    innersum = 0
    for iI in range(d):
        innersum += c[iI]**2 * (x[iI] - w[iI])**2
    return np.exp(- innersum)

def corner_peak(x, c, w, d):
    innersum = 0
    for iI in range(d):
        innersum +=  c[iI] * x[iI]
    return (1 + innersum) ** -(d+1)

# Sparse Grids
def sparse_grid(iDim, iOut, which_bases, func, c, w):

    grid  = TasmanianSG.TasmanianSparseGrid()

    depths  = np.arange(3, 10, 1)
    n = 1000
    errorStore = np.zeros(len(depths))
    numPoints = np.zeros(len(depths))

    for index, iDepth in enumerate(depths):
        # n iDim-dimensional sample points
        aPnts = np.empty((n, iDim))
        for iI in range(n):
            for iJ in range(2):
                aPnts[iI][iJ] = uniform(-1.0, 1.0)

        # Result
        aTres = np.empty([n,])
        for iI in range(n):
            aTres[iI] = func(aPnts[iI], c, w, iDim)

        print("\n-------------------------------------------------------------------------------------------------")
        if func == gaussian:
            print("Test Function: interpolate function f(x,y) = exp( -sum_i c_i^2(x_i - w_i)^2 )")
        if func == oscillatory:
            print("Test Function: interpolate function f(x,y) = cos( 2 * pi * w_1 + sum_i c_i * x_i )")
        if func == corner_peak:
            print("Test Function: interpolate function f(x,y) = ( 1 + sum_i c_i * x_i )^(-3)")
        print("using fixed sparse grid with depth {0:1d}".format(iDepth))
        print("the error is estimated as the maximum from 1000 random points\n")

        # Construct sparse grid
        grid.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")
        aPoints = grid.getPoints()
        iNumP1 = aPoints.shape[0]
        aVals = np.empty([aPoints.shape[0], 1])
        for iI in range(aPoints.shape[0]):
            aVals[iI] = func(aPoints[iI], c, w, iDim)
        grid.loadNeededPoints(aVals)

        # compute the error
        aRes = grid.evaluateBatch(aPnts)
        errorStore[index] = max(np.fabs(aRes[:,0] - aTres))
        numPoints[index] = iNumP1
        print(" For localp    Number of points: {0:1d}   Max. Error: {1:1.16e}".format(iNumP1, errorStore[index]))

        # write coordinates of grid to a text file

    f=open("errors.txt", 'a')
    np.savetxt(f,errorStore, fmt='% 2.16f')
    f.close()

    return numPoints, errorStore

# Adaptive Sparse Grids
def adaptive_sparse_grid(iDim, iOut, fTol, which_basis, refinement_level, func, c, w):

    grid1  = TasmanianSG.TasmanianSparseGrid()
    depths = np.arange(3, 8, 1)
    errorStore = np.zeros(len(depths))
    numPoints = np.zeros(len(depths))
    n = 1000

    for index, iDepth in enumerate(depths):

        aPnts = np.empty([n, 2])
        for iI in range(n):
            for iJ in range(2):
                aPnts[iI][iJ] = uniform(-1.0, 1.0)

        aTres = np.empty([n,])
        for iI in range(n):
            aTres[iI] = func(aPnts[iI], c, w, iDim)

        # level of grid before refinement
        grid1.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")

        aPoints = grid1.getPoints()
        aVals = np.empty([aPoints.shape[0], 1])
        for iI in range(aPoints.shape[0]):
            aVals[iI] = func(aPoints[iI], c, w, iDim)
        grid1.loadNeededPoints(aVals)

        print("\n-------------------------------------------------------------------------------------------------")
        if func == gaussian:
            print("Test Function: interpolate function f(x,y) = exp( -sum_i c_i^2(x_i - w_i)^2 )")
        if func == oscillatory:
            print("Test Function: interpolate function f(x,y) = cos( 2 * pi * w_1 + sum_i c_i * x_i )")
        if func == corner_peak:
            print("Test Function: interpolate function f(x,y) = ( 1 + sum_i c_i * x_i )^(-3)")
        print("   the error is estimated as the maximum from 1000 random points")
        print("   tolerance is set at 1.E-6 and piecewise linear basis functions are used\n")

        print("               Classic refinement ")
        print(" refinem lev   points    error   ")

        #refinement level
        for iK in range(refinement_level):

            grid1.setSurplusRefinement(fTol, 1, "fds")   #also use fds, or other rules
            aPoints = grid1.getNeededPoints()
            aVals = np.empty([aPoints.shape[0], 1])
            for iI in range(aPoints.shape[0]):
                aVals[iI] = func(aPoints[iI], c, w, iDim)
            grid1.loadNeededPoints(aVals)

            aRes = grid1.evaluateBatch(aPnts)
            fError1 = max(np.fabs(aRes[:,0] - aTres))

            numPoints[index] = aPoints.shape[0]
            errorStore[index] = fError1

            print(" {0:9d} {1:9d}  {2:1.2e}".format(iK+1, grid1.getNumPoints(), fError1))

    f=open("errorsASG.txt", 'a')
    np.savetxt(f,errorStore, fmt='% 2.16f')
    f.close()

    return numPoints, errorStore

# Sparse Grid with dimension 2 and 1 output and refinement level 5
iDim = 2
iOut = 1
which_basis = 1 #1= linear basis functions -> Check the manual for other options

numPointsS, errorStoreS = sparse_grid(iDim, iOut, which_basis, oscillatory, c, w)
numPointsS1, errorStoreS1 = sparse_grid(iDim, iOut, which_basis, gaussian, c, w)
numPointsS2, errorStoreS2 = sparse_grid(iDim, iOut, which_basis, corner_peak, c, w)

plt.xscale('log')
plt.yscale('log')
plt.plot(numPointsS, errorStoreS, color = 'b')
plt.plot(numPointsS1, errorStoreS1, color = 'g')
plt.plot(numPointsS2, errorStoreS2, color = 'r')
plt.plot(numPointsS, errorStoreS, 'bo', label = "oscillatory")
plt.plot(numPointsS1, errorStoreS1, 'go', label = "gaussian")
plt.plot(numPointsS2, errorStoreS2, 'ro', label = "corner peak")
plt.title('Sparse Grid Approximation')
plt.xlabel('# Points')
plt.ylabel('Max Error')
plt.legend(loc = "upper right")
plt.savefig('SG_approx.png')
plt.close()
# Adaptive Sparse Grid with dimension 2 and 1 output and maximum refinement level 5, refinement criterion.
iDim = 2
iOut = 1
fTol = 1.E-5
which_basis = 1
refinement_level = 5

numPointsAS, errorStoreAS = adaptive_sparse_grid(iDim, iOut, fTol, which_basis, refinement_level, oscillatory, c, w)
numPointsAS1, errorStoreAS1 = adaptive_sparse_grid(iDim, iOut, fTol, which_basis, refinement_level, gaussian, c, w)
numPointsAS2, errorStoreAS2 = adaptive_sparse_grid(iDim, iOut, fTol, which_basis, refinement_level, corner_peak, c, w)

plt.yscale('log')
plt.xscale('log')
plt.plot(numPointsAS, errorStoreAS, color = 'b')
plt.plot(numPointsAS1, errorStoreAS1, color = 'g')
plt.plot(numPointsAS2, errorStoreAS2, color = 'r')
plt.plot(numPointsAS, errorStoreAS, 'bo', label = "oscillatory")
plt.plot(numPointsAS1, errorStoreAS1, 'go', label = "gaussian")
plt.plot(numPointsAS2, errorStoreAS2, 'ro', label = "corner peak")
plt.title('Adaptive Sparse Grid Approximation')
plt.xlabel('# Points')
plt.ylabel('Max Error')
plt.legend(loc = "upper right")
plt.savefig('adaptive_SG_approx.png')
