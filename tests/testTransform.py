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
from utils import buildBlocks, buildBCHOL

# Constants
FILE_TYPE = "json"
INIT = False


class TestTransformFromJson(unittest.TestCase):

    def setUp(self):
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

    def check_transform(self, Q, R, q, r, A, B, d, N, nx, nu):
        """Check the Blocks to Bchol transformation and assert values."""
        G, g, C, c = buildBlocks(N, nu, nx, Q, R, q, r, A, B, d)

        # Test the return values from buildBCHOL
        Q_test, R_test, q_test, r_test, A_test, B_test, d_test = buildBCHOL(G, g, C, c, N, nx, nu)

        self.assertTrue(np.isclose(Q, Q_test).all(), "Q and Q_test are not close!")
        self.assertTrue(np.isclose(R, R_test).all(), "R and R_test are not close!")
        self.assertTrue(np.isclose(q, q_test).all(), "q and q_test are not close!")
        self.assertTrue(np.isclose(r, r_test).all(), "r and r_test are not close!")
        self.assertTrue(np.isclose(A, A_test).all(), "A and A_test are not close!")
        self.assertTrue(np.isclose(B, B_test).all(), "B and B_test are not close!")
        self.assertTrue(np.isclose(d, d_test).all(), "d and d_test are not close!")
        
    def test_transform_blocks(self):
        """Test transformation from LQR setup to KKT and back."""
        # Step 1: Prepare data
        Q, R, q, r, A, B, d, c, N, nx, nu = self.prepare_matrices(self.N, self.lqrdata)

        # Step 2: Build blocks and compare results
        self.check_transform(Q, R, q, r, A, B, d, N, nx, nu)

        # Step 3: Print success
        print("Passed blocks to LQR correctly\n")

    def test_correctBlocks(self):
        # Step 1: Prepare data
        Q, R, q, r, A, B, d, c, N, nx, nu = self.prepare_matrices(self.N, self.lqrdata)

        # Step 2: Build the blocks
        G, g, C, c = buildBlocks(N, nu, nx, Q, R, q, r, A, B, d)
        states=nu+nx
        #check that G is correctly build - includes R zero of the last tim
        for i in range(N):
            self.assertTrue(np.isclose(Q[i], G[states*i:states*i+nx,states*i:states*i+nx]).all(), "Q is not correctly placed in G block!")
            self.assertTrue(np.isclose(R[i], G[states*i+nx:states*(i+1),states*i+nx:states*(i+1)]).all(), "R is not correctly placed in G block!")
        #check that g is correctly build
        for i in range(N):
            self.assertTrue(np.isclose(q[i], g[states*i:states*i+nx]).all(), "q is not correctly placed in g block!")
            self.assertTrue(np.isclose(r[i], g[states*i+nx:states*(i+1)]).all(), "r is not correctly placed in g block!")
        #check that C is correctly build
        for i in range(N):
            self.assertTrue(np.array_equal(np.eye(nx), C[nx*i:nx*(i+1), states*i:states*i+nx]), 
                     "Identity matrix is not correctly places in C.")
            #A, and B are negated and B is transposed (I feel like only A or B should be negated)
            if(i>0):
                self.assertTrue(np.array_equal(-A[i-1], C[nx*i:nx*(i+1), states*(i-1):states*(i-1)+nx]), 
                        "A matrix is not correctly placed in C.")
                self.assertTrue(np.array_equal(B[i-1].T, C[nx*i:nx*(i+1), states*i-nu:states*i]), 
                        "B matrix is not correctly placed in C.")
        #check that c is correctly build
        for i in range(N):
            self.assertTrue(np.isclose(d[i], c[i*nx:(i+1)*nx]).all(), "d is not correctly placed in c block!")



    
    def test_kkt(self):
        """Check that KKT in the structure that you'd expect"""
        # Step 1: Prepare data
        Q, R, q, r, A, B, d, c, N, nx, nu = self.prepare_matrices(self.N, self.lqrdata)

        # Step 2: Build KKT and compare results
        G, g, C, c = buildBlocks(N, nu, nx, Q, R, q, r, A, B, d)
        KKT,kkt = buildKKT(N,nu, nx,Q,R,q,r,A,B,d)
        states=nu+nx
        #check that G is in the right place
        self.assertTrue(np.array_equal(G[:-nu,:-nu],KKT[:(N-1)*states+nx,:(N-1)*states+nx]), 
                        "G matrix is not correctly placed in kkt.")
        #check that C is in the right place
        self.assertTrue(np.array_equal(C,KKT[(N-1)*states+nx:,:(N-1)*states+nx]), 
                        "C matrix is not correctly placed in kkt.")

        #check transposed C
        self.assertTrue(np.array_equal(C.T,KKT[:(N-1)*states+nx,(N-1)*states+nx:]), 
                        "C.T matrix is not correctly placed in kkt.")

        #check all zeroes
        self.assertTrue(np.all(KKT[(N-1)*states+nx:,(N-1)*states+nx:]==0), 
                        "Zero matrix is not correctly placed in kkt.")

        #Check the vector
        # breakpoint()
        #Check that g is in the right place
        self.assertTrue(np.array_equal(g[:-nu],kkt[:(N-1)*states+nx]), 
                        "g matrix is not correctly placed in kkt.")


        #Check c is in the right place
        self.assertTrue(np.array_equal(c,kkt[(N-1)*states+nx:]), 
                        "g matrix is not correctly placed in kkt.")

    # def test_kkt(self):
    #     """Check KKT gives the same answer via np.linalg.solve"""


    


if __name__ == '__main__':
    unittest.main()

