import numpy as np
import math
import copy
import scipy.linalg as linalg
from scipy.linalg.blas import dgemv,dgemm
"""
Call this function to build KKT matrix and kkt vector 
"""
def buildKKT(N,nu, nx,Q,R,q,r,A,B,d,
             C_test,c_test,G_test,g_test,KKT_test): 
    assert N==len(Q)
    assert nx==Q[0].shape[1]
    assert nu==R[0].shape[1] 
    n = nx + nu
    

    ####Check the dim
    dim = N*(nx*2+nu)
    KKT=np.zeros((dim,dim)) 
    #####
    #build G
    G=np.zeros((N*(nx+nu),N*(nx+nu)))
    for i in range(N):
        qi=i*(nx+nu)
        ri=qi+nx
        G[qi:qi+nx,qi:qi+nx]=Q[i]
        G[ri:ri+nu,ri:ri+nu]=R[i]

    #build g
    g_interleaved = []
    for i in range(N):
    # Combine each row of q and r
        combined_row = np.hstack([q[i], r[i]])
        g_interleaved.append(combined_row)
    g_reshaped = np.array(g_interleaved)
    g= g_reshaped.flatten()

    #rebuild C - CHECKED_ CORRECT!
    C = np.zeros((N*nx,N*n))
    #check if you need to negate
    A=-A
    B=-B
    B=B.transpose(0,2,1)
    for i in range(N-1):
        row = nx+i*nx
        col =i*(nx+nu)
        C[row:row+nx,col:col+nx]=A[i]
        if(nu==1):
            C[row:row+nx,col+nx]=B[i].flatten()
        else:
            C[row:row+nx,col+nx]=B[i]    

    #add identitiy matrix
    for i in range(N):
         row =i*nx
         col=i*(nx+nu)
         C[row:row+nx, col:col+nx]=np.eye(nx)
    close_elementsC = np.isclose(C[:,:-1], C_test, rtol=1e-5, atol=1e-8)
    if np.allclose(C[:,:-1], C_test):
         print("C correct")
    else:
         print("C WRONG")
         breakpoint()

    if np.allclose(G[:-1,:-1], G_test):
         print("G correct")
    else:
         print("G WRONG")
         breakpoint()

    c=d.flatten()
    BR = np.zeros((nx*N,nx*N))
    C=C[:,:-1]
    KKT = np.hstack((np.vstack((G[:-1,:-1], C)),np.vstack((C.transpose(), BR))))

    if np.allclose(KKT, KKT_test):
         print("KKT new ok!")
    else:
        print("WRONG KKT!!")

    kkt = np.concatenate((g, c))

    print("BUILT KKT!!")
    return KKT,kkt

    breakpoint()