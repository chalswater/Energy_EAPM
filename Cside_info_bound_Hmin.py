#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:34:42 2024

@author: carles
"""
import numpy as np
import cvxpy as cp
from cvxpy import *
import time
from MoMPy.MoM import *

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
id_2 = np.eye(2)

def W322_qubits(rho,nX,nB,nY,dim):
    
    """ Optimal correlations obtainable in a sequential measurement """
    
    # --------------
    # Variables
    # --------------
    
    M = {}
    for y in range(nY):
        M[y] = {}
        for b in range(nB):
            M[y][b] = cp.Variable((dim,dim),complex=True)
            
    # --------------
    # Constraints
    # --------------
    
    ct = []
    
    ct += [ M[y][b] >> 0.0 for b in range(nB) for y in range(nY) ]
    ct += [ M[y][b] == M[y][b].H for b in range(nB) for y in range(nY) ]
    ct += [ sum([ M[y][b] for b in range(nB) ]) == np.identity(dim) for y in range(nY) ]

    E = {}
    for x in range(nX):
        E[x] = {}
        for y in range(nY):
            E[x][y] = cp.trace(rho[x] @ M[y][0]) - cp.trace(rho[x] @ M[y][1])

    W322 = cp.real( E[0][0] + E[0][1] + E[1][0] - E[1][1] - E[2][0] )

    # --------------
    # Run the SDP
    # --------------
    
    obj = cp.Maximize(W322)
    prob = cp.Problem(obj,ct)
    
    output = []
    
    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)
    
    except SolverError:
        something = 10
        
    # --------------
    # Output
    # --------------
    
    output = []
    
    output += [W322.value]
    
    Mout = {}
    for y in range(nY):
        Mout[y] = {}
        for b in range(nB):
            Mout[y][b] = M[y][b].value
            
    output += [ Mout ]
    
    return output

def max_ps(rho,nX,nY,nB,dim,Q):
    
    """ Optimal correlations obtainable in a sequential measurement """
    
    # --------------
    # Variables
    # --------------
    
    M = {}
    for y in range(nY):
        M[y] = {}
        for b in range(nB):
            M[y][b] = cp.Variable((dim,dim),complex=True)
                
    # --------------
    # Constraints
    # --------------
    
    ct = []
    
    ct += [ M[y][b] >> 0.0 for b in range(nB) for y in range(nY) ]
    ct += [ M[y][b] == M[y][b].H for b in range(nB) for y in range(nY) ]
    ct += [ sum([ M[y][b] for b in range(nB) ]) == np.identity(dim) for y in range(nY) ]
    
    ct += [ cp.real( sum([ cp.trace( M[y][b] @ rho[x] )/(nX*nY) for b in range(nX,nB) for x in range(nX) for y in range(nY) ]) ) >= Q  ]

    ps = cp.real( sum([ cp.trace( M[y][x] @ rho[x] )/(nX*nY) for x in range(nX) for y in range(nY) ]))

    # --------------
    # Run the SDP
    # --------------

    obj = cp.Maximize(ps)
    prob = cp.Problem(obj,ct)
    
    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)
    
    except SolverError:
        something = 10
        
    
    # --------------
    # Output
    # --------------
    
    output = []
    output += [ps.value]
    
    Mout = {}
    for y in range(nY):
        Mout[y] = {}
        for b in range(nB):
            Mout[y][b] = M[y][b].value
            
    output += [Mout]
    
    return output

def Hmin_CSI_sDI(nX,nY,nB,nK,energy,monomials,gamma_matrix_els,xstar,ystar,W):
    
    [w_R,w_B,w_S] = monomials
    [G_new,map_table,S,list_of_eq_indices,Mexp] = gamma_matrix_els

    nL = nB

    #----------------------------------------------------#
    #                  CREATE VARIABLES                  #
    #----------------------------------------------------#
    
    G_var_vec = {}
    for l in range(nL):
        G_var_vec[l] = {}
        for element in list_of_eq_indices:
            if element == map_table[-1][-1]:
                G_var_vec[l][element] = 0.0 # Zeros form orthogonal projectors
            else:
                G_var_vec[l][element] = cp.Variable()
                
    q = cp.Variable(nL,nonneg=True)
            
    #--------------------------------------------------#
    #                  BUILD MATRICES                  #
    #--------------------------------------------------#
    
    G = {}
    for l in range(nL):
        lis = []
        for r in range(len(G_new)):
            lis += [[]]
            for c in range(len(G_new)):
                lis[r] += [G_var_vec[l][G_new[r][c]]]
        G[l] = cp.bmat(lis)

    #------------------------------------------------------#
    #                  CREATE CONSTRAINTS                  #
    #------------------------------------------------------#
    
    ct = []
    
    ct += [ G[l] >> 0.0 for l in range(nL) ]
    
    ct += [ sum([ q[l] for l in range(nL) ]) == 1.0 ]
    
    ct += [ G_var_vec[l][fmap(map_table,[0])] == q[l]*2 for l in range(nL) ]
    
    ct += [ G_var_vec[l][fmap(map_table,[w_R[x]])] == q[l] for x in range(nX) for l in range(nL) ]
    ct += [ G_var_vec[l][fmap(map_table,[w_S[k]])] == q[l] for k in range(nK) for l in range(nL) ]
    
    ct += [ G_var_vec[l][fmap(map_table,[w_B[y][b]])] == q[l] for y in range(nY) for b in range(nB) for l in range(nL) ]
    
    # Witness or full distribution
    if W == None:
        ct += [ sum([ G_var_vec[l][fmap(map_table,[w_R[x],w_B[y][b]])] for l in range(nL) ]) == pbxy[b][x][y] for b in range(nB) for x in range(nX) for y in range(nY) ]
    else:
        ct += [ sum([ G_var_vec[l][fmap(map_table,[w_R[x],w_B[y][x]])]/float(nX) for x in range(nX) for l in range(nL) ]) >= W for y in range(nY) ]
        
    # For sigma being PN projectors
    ct += [ sum([ G_var_vec[l][fmap(map_table,[w_R[x],w_S[0]])] for l in range(nL) ]) == 1.0-energy for x in range(nX) ]
    
    goal = sum([ G_var_vec[b][fmap(map_table,[w_R[xstar],w_B[ystar][b]])] for b in range(nB) ])
    
    #----------------------------------------------------------------#
    #                  RUN THE SDP and WRITE OUTPUT                  #
    #----------------------------------------------------------------#

    obj = cp.Maximize(goal)
    prob = cp.Problem(obj,ct)

    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

    except SolverError:
        something = 10
    
    if goal.value != None:
        return -np.log2(goal.value)
    else:
        return None


#------------------------------------------------------------------------------------------

def prepare_hierarchy_HminC_SDI(nX,nY,nB,nK,some_second):

    # Track operators in the tracial matrix
    w_R = [] # State
    w_B = [] # Bob
    w_S = [] # Observable

    S_1 = [] # List of first order elements
    cc = 1

    for x in range(nX):
        S_1 += [cc]
        w_R += [cc]
        cc += 1

    for y in range(nY):
        w_B += [[]]
        for b in range(nB):
            S_1 += [cc]
            w_B[y] += [cc]
            cc += 1
            
    for k in range(nK):
        S_1 += [cc]
        w_S += [cc]
        cc += 1

    # Additional higher order elements
    S_high = [] # Uncomment if we only allow up to some 2nd order elements in the hierarchy  

    # Second order elements
    if some_second == True:
        S_2 = []

        for b in range(nB):  
            for y in range(nY):
                for x in range(nX):
                    S_high += [[w_B[y][b],w_R[x]]]
                 
        for x in range(nX):     
             for xx in range(nX):
                 S_high += [[w_R[xx],w_R[x]]]
                 
        for x in range(nX):     
            S_high += [[w_R[x],w_S[0]]]
    else:
        S_2 = S_1
        
        # for x in range(nX):     
        #      for xx in range(nX):
        #          for xxx in range(nX):
        #              S_high += [[w_R[xxx],w_R[xx],w_R[x]]]
                     
        #          for k in range(nK):
        #              S_high += [[w_S[k],w_R[xx],w_R[x]]]
        #              S_high += [[w_R[xx],w_S[k],w_R[x]]]
                     
        #      for k in range(nK):
        #          for kk in range(nK):
        #              S_high += [[w_S[kk],w_R[x],w_S[k]]]
                     
        # for b in range(nB):  
        #     for y in range(nY):
        #         for x in range(nX):
        #             for k in range(nK):
        #                 S_high += [[w_B[y][b],w_R[x],w_S[k]]]
        #                 S_high += [[w_B[y][b],w_S[k],w_R[x]]]
        #                 S_high += [[w_S[k],w_B[y][b],w_R[x]]]
                        
        # for b in range(nB):  
        #     for y in range(nY):
        #         for x in range(nX):
        #             for xx in range(nX):
        #                 S_high += [[w_B[y][b],w_R[x],w_R[xx]]]
        #                 S_high += [[w_R[x],w_B[y][b],w_R[xx]]]
                        
                        
        # for b in range(nB):
        #     for y in range(nY):
        #         for bb in range(nB):
        #             for yy in range(nY):
        #                 for x in range(nX):
        #                     S_high += [[w_B[y][b],w_R[x],w_B[yy][bb]]]
                            
    # Set the operational rules within the SDP relaxation
    list_states = [] # operators that do not commute with anything (not important here)

    rank_1_projectors = []
    rank_1_projectors += [ w_R[x] for x in range(nX) ]
    rank_1_projectors += [ w_B[y][b] for b in range(nB) for y in range(nY) ] 
    rank_1_projectors += [ w_S[k] for k in range(nK) ]

    orthogonal_projectors = []
    orthogonal_projectors += [ [ w_B[y][b] for b in range(nB) ] for y in range(nY) ] 
    orthogonal_projectors += [ [ w_S[k] for k in range(nK) ] ]

    commuting_variables = [] # commuting elements (wxcept with elements in "list_states"

    print('Rank-1 projectors',rank_1_projectors)
    print('Orthogonal projectors',orthogonal_projectors)
    print('commuting elements',commuting_variables)

    # Collect rules and generate SDP relaxation matrix
    start = time.process_time()
    [G_new,map_table,S,list_of_eq_indices,Mexp] = MomentMatrix(S_1,S_2,S_high,rank_1_projectors,orthogonal_projectors,commuting_variables,list_states)
    end = time.process_time()

    print('Gamma matrix generated in',end-start,'s')
    print('Matrix size:',np.shape(G_new))

    monomials = [w_R,w_B,w_S]
    gamma_matrix_els = [G_new,map_table,S,list_of_eq_indices,Mexp]
    
    return monomials, gamma_matrix_els

#------------------------------------------------------------------------------------------

nX = 2
nY = 1
nB = 2
nK = 1

xstar = 0
ystar = 0

N = 100 # number of datapoints

energy = 0.005

some_second = False

vec = np.linspace(psN(nX,energy)*0.91,psN(nX,energy),N)
# vec = np.linspace(1e-4,0.5,N)

HminC_SDI_vec = [[],[]]

monomials, gamma_matrix_els = prepare_hierarchy_HminC_SDI(nX,nY,nB,nK,some_second)

for ii in range(N):

    # energy = vec[ii]#psN(nX,energy)
    Wobs = vec[ii]
    
    start = time.process_time()
    out_C = Hmin_CSI_sDI(nX,nY,nB,nK,energy,monomials,gamma_matrix_els,0,0,Wobs)
    end = time.process_time()
    
    HminC_SDI_vec[0] += [vec[ii]]
    HminC_SDI_vec[1] += [out_C]
        
    print(out_C, 'in',np.round(end-start,2),'seconds')
    
    
    
    
    
    
    
    
    
    





