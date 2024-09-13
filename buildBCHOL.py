#general imports

import pdb
import copy
import scipy
import sys
import math
import numpy as np
import json
import csv
INIT = True

from . import BCHOL
from . import buildKKT



def buildBCHOL(G: np.ndarray, g: np.ndarray, C: np.ndarray, c: np.ndarray, N: int,nx: int, nu: int,KKT,kkt):


    """
    Prepares the matrices to the right format in order to launch LQR kerne;

    Parameters:
    G (np.ndarray): A matrix of Q_R from pendulum problem .
    g (np.ndarray): A combined vector of q_r
    C (np.ndarray): A matrix of A_B and I, negated
    c (np.ndarray): A d vector

    Returns:
    CHECK with Brian what it needs to return
    """    """
    runSolve is the main interface that lets you execute all the
    calculations to solve the LQR problem.
    """

    #extract Q, R from G - CORRECT
    Q_list=[]
    R_list=[]
    for i in range(N):
         if(i!=N-1):
              qi=i*(nx+nu)
              ri=qi+nx
              Q_temp=G[qi:qi+nx,qi:qi+nx]
              R_temp = G[ri:ri+nu,ri:ri+nu]
              Q_list.append(Q_temp)
              R_list.append(R_temp)
         else:
            qi=i*(nx+nu)
            ri=qi+nx
            Q_temp=G[qi:qi+nx,qi:qi+nx]
            R_temp = np.zeros((nu,nu))
            Q_list.append(Q_temp)
            R_list.append(R_temp)
    Q=np.array(Q_list)
    R=np.array(R_list)

   #build G - checking buildKKT
#     G1=np.zeros((N*(nx+nu),N*(nx+nu)))
#     for i in range(N):
#         qi=i*(nx+nu)
#         ri=qi+nx
#         G1[qi:qi+nx,qi:qi+nx]=Q[i]
#         G1[ri:ri+nu,ri:ri+nu]=R[i]
#     G1 = G1[:-1,:-1]
#     close_elementsG = np.isclose(G, G1, rtol=1e-5, atol=1e-8)
#     if np.allclose(G, G1):
#          print("G correct")
#     else:
#          print("WRONG")
#          breakpoint()



    #preparing q_r as separate vector, add r=0 at the last timestep
    g=np.append(g,np.zeros(nu))
    g_reshaped = g.reshape(-1, nx+nu)
    q = g_reshaped[:, :nx].flatten()
    q =q.reshape(-1,nx)
    #extract r from g
    r = g_reshaped[:,-1].flatten()
    r=r.reshape(-1,1)

#     #build g
#     g_interleaved = []
#     for i in range(N):
#     # Combine each row of q and r
#         combined_row = np.hstack([q[i], r[i]])
#         g_interleaved.append(combined_row)
#     g_reshaped = np.array(g_interleaved)
#     g1= g_reshaped.flatten()
#     #check g/g1
#     close_elementsG = np.isclose(g, g1, rtol=1e-5, atol=1e-8)
#     if np.allclose(g, g1):
#          print("g correct")
#     else:
#          print("WRONG")
#          breakpoint()
    
    


    #get A,B from C
    A_list =[] 
    B_list =[]
    for i in range (N-1):
            row = nx+i*nx
            col =i*(nx+nu)
            A_temp = C[row:row+nx,col:col+nx]
            B_temp = C[row:row+nx,col+nx]
            A_list.append(A_temp)
            B_list.append(B_temp)
    A = np.array(A_list) 
    B = np.array(B_list)
    B=B.reshape(N-1,2,1)
    #transpose B
    B = B.transpose(0, nx, nu)
    #add 0s at the last timestep
    A = np.concatenate((A,np.zeros((1,nx,nx))),axis=0)
    B=np.concatenate((B,np.zeros((1,nu,nx))),axis=0)
    #negate both A and B 
    A=-A
    B=-B

#rebuild C
#     C1 = np.zeros((N*nx,N*(nx+nu)))
#     #check if you need to negate
#     A1=-A
#     B1=-B
#     B1=B1.transpose(0,2,1)
#     for i in range(N-1):
#         row = nx+i*nx
#         col =i*(nx+nu)
#         C1[row:row+nx,col:col+nx]=A1[i]
#         C1[row:row+nx,col+nx]=B1[i].flatten()
 
#     #add identitiy matrix
#     for i in range(N):
#          row =i*nx
#          col=i*(nx+nu)
#          C1[row:row+nx, col:col+nx]=np.eye(nx)
#     C1 = C1[:,:-1]
#     close_elementsC = np.isclose(C, C1, rtol=1e-5, atol=1e-8)
#     if np.allclose(C, C1):
#          print("C correct")
#     else:
#          print("C WRONG")
#          breakpoint()
   ##SEEMS LIKE EVERYTHING IS CORRECR BESIDES KKT
    BR = np.zeros((N*nx,N*nx))
    

#     KKT1 = np.hstack((np.vstack((G1, C1)),np.vstack((C.transpose(), BR))))
#     KKT_new =np.hstack((np.vstack((G, C)),np.vstack((C.transpose(), BR))))
#     if np.allclose(KKT, KKT_new):
#          print("KKT new ok!")
#     if np.allclose(KKT1, KKT):
#          print("KKT1 new ok!")
    #here all good

    #get d (just copy c vector)
    d=c.reshape(-1,2)

    #check that you can reconstruct KKT
#     KKT2,kkt2 = buildKKT.buildKKT(N,nu, nx,Q,R,q,r,A,B,d)
    KKT2,kkt2 = buildKKT.buildKKT(N,nu, nx,Q,R,q,r,A,B,d,C,c,G,g,KKT)

    close_elements = np.isclose(KKT, KKT2, rtol=1e-5, atol=1e-8)

    if np.allclose(KKT[0], KKT2[0]):
         print("DONE!")
    else:
         print("WRONG")
         not_close_indices = ~np.isclose(KKT[0], KKT2[0][:-1])

    # Print the indices and the corresponding values in both arrays
         print("Indices where elements are not close:")
         for idx, (val1, val2) in enumerate(zip(KKT[0], KKT2[0][:-1])):
               if not_close_indices[idx]:
                    print(f"Index: {idx}, KKT[0]: {val1}, KKT2[0]: {val2}")

    breakpoint()


    BCHOL(N,nu,nx,Q,R,q,r,A,B,d)
    print("soln:\n")
    for i in range(N):       
        print(f"d_{i} {d[i]}") #lambdas
        print(f"q_{i}  {q[i]}") #x vector
        print(f"r_{i} {r[i]}") #u vector
