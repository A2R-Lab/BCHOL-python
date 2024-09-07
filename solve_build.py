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

#function specific imports
from . import BCHOL


#check if they are lists or np.arrays
def buildBCHOL(G: np.ndarray, g: np.ndarray, C: np.ndarray, c: np.ndarray, N: int,nx: int, nu: int):


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

    #compare G1 to G
    epsilon= 1e-5
    if(np.allclose(G, G1[:g.size,:g.size], atol=epsilon)):
         print("G reconstrucred succefully\n")
    else:
         print("DIFFERENT!\n")

    #preparing q_r as separate vector, add r=0 at the last timestep
    g=np.append(g,np.zeros(nu))
    g_reshaped = g.reshape(-1, nx+nu)
    q = g_reshaped[:, :nx].flatten()
    q =q.reshape(-1,nx)
    #extract r from g
    r = g_reshaped[:,-1].flatten()
    r=r.reshape(-1,nu)
    print("q ",q)
    print("r ",r)
    
    

    #check g reconstuction
    g_interleaved = []
    for i in range(N):
    # Combine each row of q and r
        combined_row = np.hstack([q[i], r[i]])
        g_interleaved.append(combined_row)
    g_reshaped = np.array(g_interleaved)

    g1 = g_reshaped.flatten()
    

    if(np.allclose(g1, g, atol=epsilon)):
         print("g reconstrucred succefully\n")
    else:
         print("DIFFERENT!\n")

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
    B=B.reshape(N-1,nx,nu)
   

    #transpose B
    B = B.transpose(0, nx, nu)
    #add 0s at the last timestep
    A = np.concatenate((A,np.zeros((1,nx,nx))),axis=0)
    B=np.concatenate((B,np.zeros((1,nu,nx))),axis=0)
    #negate both A and B (do I need to do it?)
    A=-A
    B=-B


    #check C - reconstruct A
    C1 = np.zeros((N*nx+nx,N*(nx+nu)))
    A=-A
    B=-B
    B=B.transpose(0,nx,nu)
    for i in range(N-1):
        row = nx+i*nx
        col =i*(nx+nu)
        C1[row:row+nx,col:col+nx]=A[i]
        C1[row:row+nx,col+nx]=B[i].flatten()
    #add identitiy matrix
    for i in range(N):
         row =i*nx
         col=i*(nx+nu)
         C1[row:row+nx, col:col+nx]=np.eye(nx)
    if(np.allclose(C1[:-2,:-1], C, atol=epsilon)):
         print("C reconstrucred succefully\n")
    else:
         print("DIFFERENT!\n")
    
    #get d (just copy c vector)
    d=c[:]
    d=d.reshape(-1,2)
    print("d shape", d.shape)
    print("c shape", c.shape)

    breakpoint()


    BCHOL(N,nu,nx,Q,R,q,r,A,B,d)
    print("soln:\n")
#     for i in range(N):       
#         print(f"d_{i} {d[i]}") #lambdas
#         print(f"q_{i}  {q[i]}") #x vector
#         print(f"r_{i} {r[i]}") #u vector
   
    # return q,r
    #construct dxul vector