

import nonlinear_solver_initial as solver     #solves opt. problems for terminal VF
import nonlinear_solver_iterate as solviter   #solves opt. problems during VFI
from parameters import *                      #parameters of model
import interpolation as interpol              #interface to sparse grid library/terminal VF
import interpolation_iter as interpol_iter    #interface to sparse grid library/iteration
import postprocessing as post                 #computes the L2 and Linfinity error of the model

import interpolation_iter_adap as interpol_iter_adap

import TasmanianSG                            #sparse grid library
import numpy as np


#======================================================================
# Start with Value Function Iteration

# terminal value function
storeVal = []
if (numstart==0):
    for sIdx in range(5):
        valnew=TasmanianSG.TasmanianSparseGrid()
        valnew=interpol.sparse_grid(n_agents, iDepth, sIdx)
        #print(valnew)
        valnew.write("valnew_1." + str(numstart) + ".txt") #write file to disk for restart
        storeVal.append(valnew)

# value function during iteration
else:
    valnew.read("valnew_1." + str(numstart) + ".txt")  #write file to disk for restart

storeValOld = []
for sIdx in range(5):
    valold=TasmanianSG.TasmanianSparseGrid()
    valold=storeVal[sIdx]
    storeValOld.append(valold)

for i in range(numstart, numits):
    storeVal = []
    for sIdx in range(5):
        #pass
        valnew=TasmanianSG.TasmanianSparseGrid()
        #valnew=interpol_iter.sparse_grid_iter(n_agents, iDepth, valold)
        valnew=interpol_iter_adap.ad_grid_iter(n_agents, iDepth, storeValOld)
        # valold=TasmanianSG.TasmanianSparseGrid()
        # valold=valnew
        # valnew.write("valnew_1." + str(i+1) + ".txt")
        storeVal.append(valnew)
    storeValOld = []
    for sIdx in range(5):
        valold=TasmanianSG.TasmanianSparseGrid()
        valold=storeVal[sIdx]
        storeValOld.append(valold)

#======================================================================
print "==============================================================="
print " "
print " Computation of a growth model of dimension ", n_agents ," finished after ", numits, " steps"
print " "
print "==============================================================="
#======================================================================

# compute errors
avg_err=post.ls_error(n_agents, numstart, numits, No_samples)

#======================================================================
print "==============================================================="
print " "
print " Errors are computed -- see errors.txt"
print " "
print "==============================================================="
#======================================================================
