import numpy as np
import math
import copy
import scipy.linalg as linalg
from scipy.linalg.blas import dgemv,dgemm
from utils import *

"""
Call this function to build KKT matrix and kkt vector 
"""
def buildKKT(N,nu, nx,Q,R,q,r,A,B,d,neg): 
    assert N==len(Q)
    assert nx==Q[0].shape[1]
    assert nu==R[0].shape[1] 
    n = nx + nu
    ####Check the di
    dim = N*(nx*2+nu)
    KKT=np.zeros((dim-1,dim-1)) 

    G,g,C,c = buildBlocks(N,nu,nx,Q,R,q,r,A,B,d)
    BR = np.zeros((nx*N,nx*N))
    C=C[:,:]
    KKT = np.hstack((np.vstack((G[:-nu,:-nu], C)),np.vstack((C.transpose(), BR))))
    g=g[:-nu]
    kkt = np.concatenate((g, c))
    kkt = kkt.reshape(dim-nu,1)


    dxul = np.linalg.solve(KKT, kkt)
    # print("OFFICIAL SOLN!")
    # with np.printoptions(precision=4, suppress=True):
    #     print(dxul)

    # print("BUILT KKT!!")
    return KKT,kkt