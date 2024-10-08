# General imports
import unittest
import json
import numpy as np
import math
import sys
import os

# Third-party libraries
import scipy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Function-specific imports
from buildKKT import buildKKT
from utils import *
from BCHOL import BCHOL


# Constants
FILE_TYPE = "json"

class TestSolveEq(unittest.TestCase):

    
    def setUp(self):
        print("setup\n")
        # Setup can load data and other initializations
        self.file_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lqr_prob.json")
        self.N, self.x0, self.lqrdata, self.soln = self.load_data(self.file_name)
    
    def load_data(self, file_name):
        """Load JSON data from a file."""
        with open(file_name, 'r') as file:
            data = json.load(file)
            N = data['nhorizon']
            x0 = data['x0']
            lqrdata = data['lqrdata']
            soln = np.array(data['soln']).flatten()
        return N, x0, lqrdata, soln

    def prepare_matrices(self, N, lqrdata):
        """Transform lists into numpy arrays and prepare matrices."""
        Q_list, R_list, q_list, r_list, A_list, B_list, d_list, c_list = [], [], [], [], [], [], [self.x0], []

        for lqr in lqrdata:
            Q_list.append(lqr['Q'])
            R_list.append(lqr['R'])
            q_list.append(lqr['q'])
            r_list.append(lqr['r'])
            c_list.append(lqr['c'])
            A_list.append(lqr['A'])
            B_list.append(lqr['B'])
            if lqr['index'] != N:
                d_list.append(lqr['d'])

        Q = np.array([np.diag(row) for row in Q_list])
        R = np.array([np.diag(row) for row in R_list])
        q = np.array(q_list)
        r = np.array(r_list)
        A = np.array(A_list)
        B = np.array(B_list)
        d = np.array(d_list)
        c = np.array(c_list)
        N = int(N)
        nx = int(lqrdata[0]['nstates'])
        nu = int(lqrdata[0]['ninputs'])

        return Q, R, q, r, A, B, d, c, N, nx, nu
    
    #checking the solution vector after solveleaf
    def check_vector_solveLeaf(self,N,q,r,d,qt,rt,dt,Qt,Rt,nx,nu):
        checkq=np.zeros((N,nx))
        checkr=np.zeros((N,nu))
        for ind in range(N):
            if (ind>0 and ind!=N-1):
                checkq[ind] = np.linalg.solve(Qt[ind],-qt[ind])
                checkr[ind] = np.linalg.solve(Rt[ind],-rt[ind])
                self.assertTrue(np.isclose(d[ind],-dt[ind]).all(), f"d {d[ind]} is wrong at index {ind}, {-dt[ind]}!")
                self.assertTrue(np.isclose(q[ind],checkq[ind]).all(), f"q is wrong at index {ind} !")
                self.assertTrue(np.isclose(r[ind],checkr[ind]).all(), f"r is wrong at index {ind}!")

    #checking F_state, and F_input after solveLeaf
    def check_AB_solveLeaf(self,N,Q,R,A,B,F_state,F_input,depth,nx,nu):
        checkState = np.zeros((N*depth,nx,nx))
        checkInput = np.zeros((N*depth,nu,nx))
        #check for N timesteps
        # breakpoint()
        checkState[2]=np.linalg.solve(Q[2],A[2])
        checkInput[2]=np.linalg.solve(R[2],B[2])
        self.assertTrue(np.isclose(F_state[2],checkState[2]).all(), f"F_state\n {F_state[2]} is wrong at index {2}, {checkState[2]}!")
        self.assertTrue(np.isclose(F_input[2],checkInput[2]).all(), f"F_input\n {F_input[2]} is wrong at index {2}, {checkInput[2]}!")


    def test_solve(self):
        Qt, Rt, qt, rt, At, Bt, dt, ct, N, nx, nu = self.prepare_matrices(self.N, self.lqrdata)

        Q, R, q, r, A, B, d, c, N, nx, nu = self.prepare_matrices(self.N, self.lqrdata)
        """These lines are taken from BCHOL 9-29"""
        print("HI\n")
        depth = int(np.log2(N))
        binary_tree =initBTlevel(N)
        #negate q_r and d vectors
        q[:]=-q #state vector
        r[:]= -r # input vector/ control vector
        d[:]= -d #lambda /lagrange multiplies
        #Set F_lambda,F_state, and F_input
        F_lambda = np.zeros((N*depth,nx,nx))
        F_state = np.zeros((N*depth,nx,nx))
        F_input = np.zeros((N*depth,nu,nx))
        for ind in range(N):
            solveLeaf(binary_tree,ind, nx,N,Q,R,q,r,A,B,d,F_lambda,F_state, F_input)
        """Checking q,r,d, against np.linalg.solve"""
        self.check_vector_solveLeaf(N,q,r,d,qt,rt,dt,Qt,Rt,nx,nu)
        """Now we also need to check Q\A, R\B"""
        self.check_AB_solveLeaf(N,Q,R,A,B,F_state, F_input,depth,nx,nu)
        print("Solve leaf passed\n")

        """Test big loop"""
        for level in range (depth):
            #get the indxs for curr level
            indx_atlevel =getValuesAtLevel(binary_tree,level)

            count =len(indx_atlevel) 
            L = int(np.power(2.0,(depth-level-1)))
            cur_depth = depth-level
            upper_levels = cur_depth-1
            num_factors = N*upper_levels
            num_perblock = num_factors//L
    
            #calc inner products Bbar and bbar (to solve y in Schur)
            for b_ind in range (L):
                for t_ind in range(cur_depth):
                    ind = b_ind * cur_depth + t_ind
                    leaf = ind // cur_depth
                    upper_level = level + (ind % cur_depth)
                    lin_ind = int(np.power(2.0, level)) * (2 * leaf + 1) - 1
                    factorInnerProduct(A,B, F_state, F_input, F_lambda, lin_ind, upper_level, N)

            #cholesky fact for Bbar/bbar 
            for leaf in range (L):
                index = int(np.power(2.0, level)) * (2 * leaf + 1) - 1
                lin_ind = index + N * level
                if(is_choleskysafe(F_lambda[lin_ind+1])):
                    F_lambda[lin_ind+1]=linalg.cho_factor(F_lambda[lin_ind+1],lower =True)[0]
                else:
                    print(f"Can't factor Cholesky {lin_ind} :\n")
                    print(F_lambda[lin_ind])

            #solve with Chol factor for y  SHUR compliment
            for b_id in range(L):
                for t_id in range(upper_levels):
                    i = b_id*upper_levels+t_id
                    leaf = i//upper_levels
                    upper_level = level+1+(i%upper_levels)
                    lin_ind = int(np.power(2,level)*(2*leaf+1))
                    Sbar = F_lambda[(lin_ind)+N*level]
                    f = F_lambda[(lin_ind)+N*upper_level]

                    if(is_choleskysafe(Sbar)):             
                        f[:]=linalg.cho_solve((Sbar,True),f,overwrite_b=True)
                    else:
                        print("Cant sovle Chol")

            # update SHUR - update x and z compliments      
            for b_id in range(L):
                for t_id in range(num_perblock):
                    i = (b_id*4)+t_id
                    k = i//upper_levels
                    upper_level = level+1+(i%upper_levels)
                    
                    index = getIndexFromLevel(N,depth,level,k,binary_tree)
                    calc_lambda  = shouldCalcLambda(index, k,binary_tree)
                    g = k+N*upper_level
                    updateShur(F_state,F_input,F_lambda,index,k,level,upper_level,calc_lambda,N)
    #         self.check_level(self,level,F_state,F_input,F_lambda)

    # def check_level(self,level,F_state,F_input,F_lambda):
    #     if(level==0):
    #         #check y==

    #         #check x
    #         #check z

    #     if(level==1):
        
    #     if(level==2):





