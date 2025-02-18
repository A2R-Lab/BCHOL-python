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
CHECK = False

#function specific imports
from BCHOL import BCHOL
from buildKKT import buildKKT



#import solve

def runSolve():
    """
    runSolve is the main interface that lets you execute all the
    calculations to solve the LQR problem.
    """

# Prompt the user to select the file type
file_type = "json"    

# Read data based on user's choice
if file_type == 'json':
    file_name = "lqr_prob.json"
    # file_name = "lqr_prob_256.json"
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
elif file_type == 'csv':
    file_name = input("Enter the CSV file name: ")
    with open(file_name,'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
        #implement the csv example later
else:
    print("Invalid file type.")

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

#INIT looks identical to the solve_lqr.cuh, 3D array use A[i] to get the matrix
#B is already transposed here 
if(INIT):
    for i in range(nhorizon):
        print("i: ",i)
        print(f"A matrix \n {A[i]}")
        print(f"B matrix: \n{B[i]}")
        print(f"Q matrix \n:{Q[i]}")
        print(f"R matrix: \n{R[i]}")
        print(f"q  {q[i]}")
        print(f"r {r[i]}")
        print(f"d {d[i]}")

#check against KKT
if (CHECK):
    KKT,kkt =buildKKT(nhorizon,ninputs, nstates,Q,R,q,r,A,B,d)
    dxul = np.linalg.solve(KKT, -kkt)
    print("Traditional KKT,np soln\n")
    with np.printoptions(precision=4, suppress=True):
        print(dxul)

#imitating calling the kernel
chol_dxul=BCHOL(nhorizon,ninputs,nstates,Q,R,q,r,A,B,d)
print("returned bchol dxul soln in the form of x,u, all lambdas later\n")
with np.printoptions(precision=4, suppress=True):
    print(chol_dxul.flatten())

print("soln as in Brian's code order:\n")
for i in range(nhorizon):
        print(f"lambda(d)_{i} {d[i]}") #lambdas
        print(f"x(q)_{i}  {q[i]}") #x vector
        print(f"u(r)_{i} {r[i]}") #u vector

#check what happens after 1 iteration

   
