import numpy as np
import math
import copy
import scipy.linalg as linalg
from scipy.linalg.blas import dgemv,dgemm
try:
    # Attempt relative import (for when the file is part of a package)
    from .utils import *
except ImportError:
    # Fallback to absolute import (for when running as a standalone script)
    from utils import *

"""
Call this function to build KKT matrix and kkt vector 
"""
def buildKKT(N,nu, nx,Q,R,q,r,A,B,d): 
    assert N==len(Q)
    assert nx==Q[0].shape[1]
    assert nu==R[0].shape[1] 
    n = nx + nu
    dim = N*(nx*2+nu)
    KKT=np.zeros((dim-1,dim-1)) 
    G,g,C,c = buildBlocks(N,nu,nx,Q,R,q,r,A,B,d)
    BR = np.zeros((nx*N,nx*N))
    C=C[:,:]
    KKT = np.hstack((np.vstack((G[:-nu,:-nu], C)),np.vstack((C.transpose(), BR))))
    g=g[:-nu]
    kkt = np.concatenate((g, c))
    dxul = np.linalg.solve(KKT, -kkt)

    return KKT,kkt

