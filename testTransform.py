#general imports

import pdb
import copy
import scipy
import sys
import math
import numpy as np
import json
import csv
INIT = False

#function specific imports
from BCHOL import BCHOL
from buildKKT import buildKKT
from utils import *



#import solve

def testTransform():
    """
    This test performs the transformation from LQR setup to KKT and back.
    Should pass all assertions to be correct.
    """
file_type = "json"    
if file_type == 'json':
    file_name = "lqr_prob.json"
    with open(file_name,'r') as file:
        data = json.load(file)
 
        # Access the top-level elements
        nhorizon = data['nhorizon']
        x0 = data['x0']
        lqrdata = data['lqrdata']
        soln = np.array(data['soln']).flatten()

        # Initialize arrays for each variable
        Q_list = []
        R_list = []
        q_list = []
        r_list = []
        c_list = []
        A_list = []
        B_list = []
        d_list = []
        d_list.append(x0)

        # Access elements inside lqrdata (assuming it's a list of dictionaries)
        for lqr in lqrdata:
            index = lqr['index']
            nstates =lqr['nstates']
            ninputs = lqr['ninputs']
            Q_list.append(lqr['Q'])
            R_list.append(lqr['R'])
            q_list.append(lqr['q'])
            r_list.append(lqr['r'])
            c_list.append(lqr['c'])
            A_list.append(lqr['A'])
            B_list.append(lqr['B'])
            if(index!=nhorizon):
                d_list.append(lqr['d'])

#transform the lists to numpy arrays
Q =np.array([np.diag(row) for row in Q_list]) 
R = np.array([np.diag(row) for row in R_list])
q = np.array(q_list)
r = np.array(r_list)
A = np.array(A_list)
B = np.array(B_list)
d = np.array(d_list)
c = np.array(c_list)
depth = int(math.log2(nhorizon))
nhorizon=int(nhorizon)
nstates= int(nstates)
ninputs=int(ninputs)

#check against KKT
KKT,kkt=buildKKT(nhorizon,ninputs, nstates,Q,R,q,r,A,B,d,False)


G,g,C,c =buildBlocks(nhorizon,ninputs, nstates,Q,R,q,r,A,B,d)
#Can we somehow check that buildBlocks is correct?

#test the return values
Q_test,R_test,q_test,r_test,A_test,B_test,d_test=buildBCHOL(G,g,C,c,nhorizon,nstates,ninputs)
print(f"{d.shape}, r shape {d_test.shape},  c shape{c.shape}, {c}\n")

#Check G,g
assert np.isclose(Q,Q_test).all(), f"Q and Q_test are not close!"
assert np.isclose(R,R_test).all(), f"R and R_test are not close!"
assert np.isclose(q,q_test).all(), f"q and q_test are not close!"
assert np.isclose(r,r_test).all(), f"r and r_test are not close!"
#Check C,c
assert np.isclose(A,A_test).all(), f"A and A_test are not close!"
assert np.isclose(B,B_test).all(), f"B and B_test are not close!"
assert np.isclose(d,d_test).all(), f"d and d_test are not close!"

print("Passed everything correctly\n")

