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
from BCHOL import BCHOL
from buildKKT import buildKKT
from utils import buildBlocks, buildBCHOL

# Constants
FILE_TYPE = "json"
INIT = False


class TestTransform(unittest.TestCase):

    def setUp(self):
        # Setup can load data and other initializations
        self.file_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lqr_prob.json")
        self.nhorizon, self.x0, self.lqrdata, self.soln = self.load_data(self.file_name)

    def load_data(self, file_name):
        """Load JSON data from a file."""
        with open(file_name, 'r') as file:
            data = json.load(file)
            nhorizon = data['nhorizon']
            x0 = data['x0']
            lqrdata = data['lqrdata']
            soln = np.array(data['soln']).flatten()
        return nhorizon, x0, lqrdata, soln


    def prepare_matrices(self, nhorizon, lqrdata):
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
            if lqr['index'] != nhorizon:
                d_list.append(lqr['d'])

        Q = np.array([np.diag(row) for row in Q_list])
        R = np.array([np.diag(row) for row in R_list])
        q = np.array(q_list)
        r = np.array(r_list)
        A = np.array(A_list)
        B = np.array(B_list)
        d = np.array(d_list)
        c = np.array(c_list)
        nhorizon = int(nhorizon)
        nstates = int(lqrdata[0]['nstates'])
        ninputs = int(lqrdata[0]['ninputs'])

        return Q, R, q, r, A, B, d, c, nhorizon, nstates, ninputs

    def check_transform(self, Q, R, q, r, A, B, d, nhorizon, nstates, ninputs):
        """Check the Blocks to Bchol transformation and assert values."""
        G, g, C, c = buildBlocks(nhorizon, ninputs, nstates, Q, R, q, r, A, B, d)

        # Test the return values from buildBCHOL
        Q_test, R_test, q_test, r_test, A_test, B_test, d_test = buildBCHOL(G, g, C, c, nhorizon, nstates, ninputs)

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
        Q, R, q, r, A, B, d, c, nhorizon, nstates, ninputs = self.prepare_matrices(self.nhorizon, self.lqrdata)

        # Step 2: Build KKT and compare results
        self.check_transform(Q, R, q, r, A, B, d, nhorizon, nstates, ninputs)

        # Step 3: Print success
        print("Passed everything correctly\n")


if __name__ == '__main__':
    unittest.main()

