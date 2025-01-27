import numpy as np
import math
import copy
import scipy.linalg as linalg
try:
    # Attempt relative import (for when the file is part of a package)
    from .utils import *
except ImportError:
    # Fallback to absolute import (for when running as a standalone script)
    from utils import *

"""
Main solver function, imitates the kernel call in CUDA.
Solves the LQR problem in a tree manner solving first the independent equations by calling the solveLeaf.
Then traversing through the levels of the binary tree and solving factorInnerProduct, cho_solve, updateShur.
First goes through the algorithm for the matrix and then doing the same for the right side vector.

Returns dxul vector.
"""
def BCHOL(knot_points,control_size, state_size,
                  Q,R,q,r,A,B,d):
  #KKT constants
  depth = int(np.log2(knot_points))
  binary_tree =initBTlevel(knot_points)

  #negate q_r and d vectors (this way we don't change the original q,r from outside of this function)
  q = np.array(-q, dtype=np.float64)
  r = np.array(-r, dtype=np.float64)
  d = np.array(-d, dtype=np.float64)



  #Set F_lambda,F_state, and F_input
  F_lambda = np.zeros((knot_points*depth, state_size, state_size), dtype=np.float64)
  F_state = np.zeros((knot_points*depth, state_size, state_size), dtype=np.float64)
  F_input = np.zeros((knot_points*depth, control_size, state_size), dtype=np.float64)


# checked that F_factors are zeroed
#   print(f"F_lambda {F_lambda}\n F_state {F_state}\n, F_input {F_input}\n") 

  for ind in range (knot_points):
      solveLeaf(binary_tree,ind, state_size,knot_points,Q,R,q,r,A,B,d,F_lambda,F_state, F_input)

 #Starting the factorization of recursive algorithm
  for level in range (depth):

      #get the indxs for curr level
      indx_atlevel =getValuesAtLevel(binary_tree,level)

      count =len(indx_atlevel) 
      L = int(np.power(2.0,(depth-level-1)))
      cur_depth = depth-level
      upper_levels = cur_depth-1
      num_factors = knot_points*upper_levels
      num_perblock = num_factors//L
 
      #calc inner products Bbar and bbar (to solve y in Schur)
      for b_ind in range (L):
         for t_ind in range(cur_depth):
            ind = b_ind * cur_depth + t_ind
            leaf = ind // cur_depth
            upper_level = level + (ind % cur_depth)
            lin_ind = int(np.power(2.0, level)) * (2 * leaf + 1) - 1
            factorInnerProduct(A,B, F_state, F_input, F_lambda, lin_ind, upper_level, knot_points)

      #cholesky fact for Bbar/bbar 
      for leaf in range (L):
         index = int(np.power(2.0, level)) * (2 * leaf + 1) - 1
         lin_ind = index + knot_points * level
         
         if(is_choleskysafe(F_lambda[lin_ind+1])):
            F_lambda[lin_ind+1]=linalg.cho_factor(F_lambda[lin_ind+1],lower =True)[0]
         else:
            print(f"Can't factor Cholesky {lin_ind} :\n")
            print(F_lambda[lin_ind])

      #solve with Chol solve for y  SHUR compliment
      for b_id in range(L):
         for t_id in range(upper_levels):
            i = b_id*upper_levels+t_id
            leaf = i//upper_levels
            upper_level = level+1+(i%upper_levels)
            lin_ind = int(np.power(2,level)*(2*leaf+1))
            Sbar = F_lambda[(lin_ind)+knot_points*level]
            f = F_lambda[(lin_ind)+knot_points*upper_level]                     
            f[:]=linalg.cho_solve((Sbar,True),f,overwrite_b=True)


   # update SHUR - update x and z compliments      
      for b_id in range(L):
         for t_id in range(num_perblock):
            i = (b_id*4)+t_id
            k = i//upper_levels
            upper_level = level+1+(i%upper_levels)
            
            index = getIndexFromLevel(knot_points,depth,level,k,binary_tree)
            calc_lambda  = shouldCalcLambda(index, k,binary_tree)
            g = k+knot_points*upper_level
            updateShur(F_state,F_input,F_lambda,index,k,level,upper_level,calc_lambda,knot_points)
  #soln vector loop, use factorized matrices for a fast solver
  for level in range (depth):
     L = int(np.power(2.0,(depth-level-1)))
     indx_atlevel = getValuesAtLevel(binary_tree,level)
     count = len(indx_atlevel)
     num_perblock = knot_points // count

   #calculate inner products with rhc - seems to be CORRECT
     for leaf in range(L):
         lin_ind = int(np.power(2,level)*(2*leaf+1)-1)
         factorInnerProduct(A,B,q,r,d,lin_ind,0,knot_points,sol=True)

   #solve for separator vars with Cached cholesky
     for leaf in range(L):
         lin_ind = int(np.power(2,level)*(2*leaf+1)-1)
         Sbar = F_lambda[level * knot_points + (lin_ind + 1)]
         zy = d[lin_ind+1]   
         zy[:]=linalg.cho_solve((Sbar,True),zy,overwrite_b=True)

     #propogate info to soln vector
     for b_id in range(L):
         for t_id in range(num_perblock):
            k = b_id * num_perblock + t_id
            index = getIndexFromLevel(knot_points,depth,level,k,binary_tree)
            calc_lambda = shouldCalcLambda(index,k,binary_tree)
            updateShur(F_state,F_input,F_lambda,index,k,level,upper_level,
                                       calc_lambda,knot_points,sol=True,d=d,q=q,r=r)

#construct dxul
  dxul = np.zeros(knot_points*(state_size*2+control_size))
#first add all x,u (q,r)
  lambda_st = (knot_points-1)*(state_size+control_size)+state_size
  for i in range(knot_points):
        start = i*(state_size+control_size)
        dxul[start:start+state_size] = q[i]
        start+=state_size
        if(i!=knot_points-1):
         dxul[start:start+control_size] = r[i]
        l_start = lambda_st+i*state_size 
        dxul[l_start:l_start+state_size] = -d[i]
  dxul=dxul[:-control_size]
  dxul=dxul.reshape(knot_points*(state_size)+(knot_points-1)*(state_size+control_size)+state_size,1)
  return dxul
