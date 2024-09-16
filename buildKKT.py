import numpy as np
import math
import copy
import scipy.linalg as linalg
from scipy.linalg.blas import dgemv,dgemm
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

    #if B is not 1 we have a problem!
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
            C[row:row+nx,col+nx:col+nx+nu]=B[i]    
    
    #add identitiy matrix
    for i in range(N):
         row =i*nx
         col=i*(nx+nu)
         C[row:row+nx, col:col+nx]=np.eye(nx)
    c=d.flatten()
    BR = np.zeros((nx*N,nx*N))
    #Get rid of the last timestep equals to 0 and build KKT,kkt
    C=C[:,:-nu]
    KKT = np.hstack((np.vstack((G[:-nu,:-nu], C)),np.vstack((C.transpose(), BR))))
    

    g=g[:-nu]
    kkt = np.concatenate((g, c))
    kkt = kkt.reshape(dim-nu,1)

    print("KKT \n")
    print(KKT)
    print("kkt\n")
    print(kkt)
    dxul = np.linalg.solve(KKT, kkt)
    print("OFFICIAL SOLN!")
    with np.printoptions(precision=4, suppress=True):
        print(dxul)

    print("BUILT KKT!!")
    return KKT,kkt