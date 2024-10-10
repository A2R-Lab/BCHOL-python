import numpy as np
import math
import copy
import scipy.linalg as linalg
from scipy.linalg.blas import dgemv,dgemm

#checked - correct
def buildBlocks(N,nu, nx,Q,R,q,r,A,B,d): 
    """
    Prepares the matrices to the right format tp build G,g,C,c blocks that will
    later be used to build KKT and be solved via SQP

    Parameters:
    N(int) : Number of timesteps
    nu(int): control
    nx(int): states
    Q: cost state matrix
    R: control matrix,
    q: cost vector
    r: control vector
    A: Dynamics matrix of states
    B: Dynamics matrix of controls
    d: affine term of dynamics


    Outputs:
    G (np.ndarray): A matrix of Q_R from pendulum problem .
    g (np.ndarray): A combined vector of q_r
    C (np.ndarray): A matrix of A_B and I, negated
    c (np.ndarray): A d vector
    """
    assert N==len(Q)
    assert nx==Q[0].shape[1]
    assert nu==R[0].shape[1] 
    n = nx + nu
    
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

    C = np.zeros((N*nx,(N-1)*n+nx))    
    A=-A
    B=B
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
    return G,g,C,c

def buildBCHOL(G: np.ndarray, g: np.ndarray, C: np.ndarray, c: np.ndarray, N: int,nx: int, nu: int):


    """
    Prepares the matrices to the right format in order to launch LQR kernel

    Parameters:
    G (np.ndarray): A matrix of Q_R from pendulum problem .
    g (np.ndarray): A combined vector of q_r
    C (np.ndarray): A matrix of A_B and I, negated
    c (np.ndarray): A d vector

    Outputs:
    N(int) : Number of timesteps
    nu(int): control
    nx(int): states
    Q: cost state matrix
    R: control matrix,
    q: cost vector
    r: control vector
    A: Dynamics matrix of states
    B: Dynamics matrix of controls
    d: affine term of dynamics
    """

    #extract Q, R from G - CORRECT
    Q_list=[]
    R_list=[]
    for i in range(N):
         if(i!=N-1):
              #row of Q
              qi=i*(nx+nu)
              #row of R
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


    #preparing q_r as separate vector, add r=0 at the last timestep
    g_reshaped = g.reshape(-1, nx+nu)
    q = g_reshaped[:, :nx].flatten()
    q =q.reshape(-1,nx)
    #extract r from g
    r = g_reshaped[:,-nu:].flatten()
    r=r.reshape(-1,nu)

    #get A,B from C
    A_list =[] 
    B_list =[]
    for i in range (N-1):
            row = nx+i*nx
            col =i*(nx+nu)
            A_temp = C[row:row+nx,col:col+nx]
            if(nu==1):
                B_temp = C[row:row+nx,col+nx]
            else:
                B_temp = C[row:row+nx,col+nx:col+nx+nu]
            A_list.append(A_temp)
            B_list.append(B_temp)
    A = np.array(A_list) 
    B = np.array(B_list)
    if(B.ndim==3):
        B=np.transpose(B, axes=(0,2,1))
    #add 0s at the last timestep
    A = np.concatenate((A,np.zeros((1,nx,nx))),axis=0)
    zeros = np.zeros((nu, B.shape[1]))
    if(B.ndim==3):
        B=np.concatenate((B,np.zeros((1,nu,nx))),axis=0)
    else:
        B=np.append(B,zeros,axis=0)
    #negate both A and B (DOUBLE CHECK!)
    A=-A
    B=B

    d=c.reshape(-1,nx)
    return Q,R,q,r,A,B,d

def buildBCHOL_KKT(N,nu, nx,Q,R,q,r,A,B,d):
    assert N==len(Q)
    assert nx==Q[0].shape[1]
    assert nu==R[0].shape[1] 
    n=nx+nu
    KKT=np.zeros(((N-1)*n+nx+(N*nx),(N-1)*n+nx+(N*nx)))
    kkt = np.zeros(((N-1)*n+nx+(N*nx)))
    #build each rearranged BCHOL block for matrix
    block_step = nx*2+nu
    for i in range (N-1):
        qi=i*block_step
        KKT[qi:qi+nx,qi:qi+nx]=Q[i]
        KKT[qi+nx:qi+nx+nu,qi+nx:qi+nx+nu]=R[i]
        ai = qi+nx+nu
        #add A to the matrix
        KKT[ai:ai+nx,qi:qi+nx]=A[i]
        #add B
        KKT[ai:ai+nx, qi+nx:qi+nx+nu] = B[i].T
        #add Transposed A
        KKT[qi:qi+nx, ai:ai+nx]=A[i].T
        #double checkB dimensions
        KKT[qi+nx:qi+nx+nu,ai:ai+nx]=B[i]
        #add -I matrix
        KKT[ai:ai+nx,qi+block_step:qi+block_step+nx]=-np.eye((nx))
        KKT[qi+block_step:qi+block_step+nx,ai:ai+nx]=-np.eye((nx))
    #for the last time step add only Q and R
    last_i = (N-1)*block_step
    KKT[last_i:last_i+nx,last_i:last_i+nx]=Q[N-1]
    KKT[last_i+nx:last_i+nx+nu,last_i+nx:last_i+nx+nu]=R[N-1]

    return KKT

def is_choleskysafe(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
    
#checked
def initBTlevel(nhorizon):
    depth = int(np.log2(nhorizon))
    levels = -np.ones(nhorizon,dtype=int)
    for level in range (depth):
        start = 2 **level-1
        step = 2 **(level+1)
        for i in range (start,nhorizon,step):
            levels[i]=level
    return levels

def  getValuesAtLevel(binarytree,level):
    index_dict = {}
    for index, value in enumerate(binarytree):
        if value not in index_dict:
            index_dict[value] = []
        index_dict[value].append(index)
    return index_dict.get(level, []) 
    
    
#correct
def solveLeaf(levels,index, nstates,nhorizon,s_Q,s_R,s_q,s_r,s_A,s_B,s_d,
              s_F_lambda,s_F_state,s_F_input):
    level = levels[index]
    lin_index = index+nhorizon*level
    #setting up array for specific indices
    A=s_A[index]
    B=s_B[index]
    Q  = s_Q[index]
    r = s_r[index]
    q = s_q[index]
    d = s_d[index]


    if(index ==0):
        R = s_R[index]
        s_F_lambda[lin_index] =np.copy(A)*-1
        s_F_input[lin_index]=np.copy(B)*1
        F_input=s_F_input[lin_index]
        R,lower_R = linalg.cho_factor(R,lower = True)
        F_input[:]=linalg.cho_solve((R,lower_R),F_input,overwrite_b=True)
        r[:]=linalg.cho_solve((R,lower_R),r,overwrite_b=True)
        #solve the block system of eqn overwriting d, q,r
        zy_temp = np.zeros(nstates)
        zy_temp=np.copy(d)*1
        d = np.copy(q)*1
        s_d[index]=dgemv(-1,Q,zy_temp,beta=-1,y=d,overwrite_y = True)
        s_q[index]=np.copy(zy_temp)*-1
        zy_temp[:] = 0
        Q,lower_Q=linalg.cho_factor(Q,lower=True)
    
    else:
        Q,lower_Q=linalg.cho_factor(Q,lower=True)
        #not the last timestep
        if(index<nhorizon-1):
            R = s_R[index]
            R,lower_R = linalg.cho_factor(R,lower =True)
            r[:]=linalg.cho_solve((R,lower_R),r, overwrite_b = True)
            s_F_state[lin_index] = np.copy(A)*1 
            F_state= s_F_state[lin_index]
            F_state[:]=linalg.cho_solve((Q,lower_Q),F_state,overwrite_b=True)
            s_F_input[lin_index] = np.copy(B)*1 
            F_input = s_F_input[lin_index]
            #CHECK IF YOU NEED TO TRANSPOSE!
            F_input[:]=linalg.cho_solve((R,lower_R),F_input,overwrite_b = True)
        
        q[:]=linalg.cho_solve((Q,lower_Q),q,overwrite_b=True)
        prev_level = levels[index-1]
        F_state_prev = s_F_state[prev_level*nhorizon+index]
        np.fill_diagonal(F_state_prev,-1)
        F_state_prev[:]=linalg.cho_solve((Q,lower_Q),F_state_prev,overwrite_b=True)
        


def factorInnerProduct(s_A,s_B, s_F_state,s_F_input,s_F_lambda,index,
                       fact_level,nhorizon,sol=False):
    C1_state=s_A[index]
    C1_input = s_B[index]
    
    if sol: 
        #perform matrix-vector multiplication if called with solution side
        F1_state = s_F_state[index]
        F1_input = s_F_input[index]
        F2_state = s_F_state[(index+1)]
        S = s_F_lambda[(index+1)]
        # Perform dgemv operations
        # S[:] = dgemv(alpha=1, a=C1_state, x=F1_state, beta=-1, y=S, trans=1)
        S = np.dot(C1_state.T, F1_state) - S
        S[:] = dgemv(alpha=1, a=C1_input.T, x=F1_input, beta=1, y=S)
        S +=-1*F2_state
        s_F_lambda[index+1]=S

    else:
        #perform matrix-matrix multiplication if called with matrix side
        lin_ind = index+(nhorizon*fact_level)
        #Dtag
        F1_state = s_F_state[lin_ind]
        F1_input = s_F_input[lin_ind]
        #Tag
        F2_state = s_F_state[(index+1)+nhorizon*fact_level]
        S = s_F_lambda[(index+1)+nhorizon*fact_level]
       
        # Perform dgemm operations
        S = np.dot(C1_state.T, F1_state) - S
        S[:] = dgemm(alpha=1, a=C1_input.T, b=F1_input, beta=1, c=S)
        S +=-1*F2_state
        s_F_lambda[(index+1)+nhorizon*fact_level] = S

def getIndexFromLevel(nhorizon,depth,level,i,levels):
    num_nodes=np.power(2,depth-level-1)
    leaf=i*num_nodes//nhorizon
    count = 0
    for k in range (nhorizon):
        if(levels[k]!=level):
            continue
        if(count==leaf):
            return k
        count+=1
    return -1

def shouldCalcLambda(index, i,levels):
    left_start = index - int(np.power(2,levels[index]))+1
    right_start = index+1
    is_start = i==left_start or i ==right_start
    return not is_start or i==0

def updateShur (s_F_state,s_F_input,s_F_lambda,index,i,level,
                upper_level,calc_lambda,nhorizon,sol = False,d=None, q = None, r=None):
    F_state = s_F_state[i+nhorizon*level]
    F_input = s_F_input[i+nhorizon*level]
    F_lambda = s_F_lambda[i+nhorizon*level]

    #for vector matrix mult
    if sol:

        f = d[index+1]
        g_state = q[i]
        g_input = r[i]
        g_lambda = d[i]
        if calc_lambda:
            g_lambda[:]=dgemv(alpha = -1,a=F_lambda, x=f,beta = 1,y=g_lambda)
        g_state[:]=dgemv(alpha=-1,a=F_state,x=f,beta=1,y=g_state)
        g_input[:]=dgemv(alpha=-1,a=F_input,x=f,beta=1,y=g_input)

    #for matrix matrix mult
    else:
        lin_index = index+1+(nhorizon*upper_level)
        f = s_F_lambda[lin_index]
        lin_index=i+nhorizon*upper_level
        g_state = s_F_state[lin_index]
        g_input = s_F_input[lin_index]
        g_lambda = s_F_lambda[lin_index]

        if calc_lambda:
            g_lambda[:]=dgemm(alpha = -1,a=F_lambda, b=f,beta = 1,c=g_lambda)
        g_state[:]=dgemm(alpha=-1,a=F_state,b=f,beta=1,c=g_state)
        g_input[:]=dgemm(alpha=-1,a=F_input,b=f,beta=1,c=g_input)

