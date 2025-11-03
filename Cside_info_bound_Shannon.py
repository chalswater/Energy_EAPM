#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:47:08 2024

@author: carles
"""

import numpy as np
import cvxpy as cp
from cvxpy import *
import chaospy
import time
from MoMPy.MoM import *

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
id_2 = np.eye(2)

def max_QRAC(rho,nX,nY,nB,dim):
    
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

    nX0 = 2
    QRAC = 0.0
    for x in range(nX):
        x1, x0 = divmod(x, nX0)
        QRAC += cp.real( cp.trace( rho[x] @ M[0][x0] ) + cp.trace( rho[x] @ M[1][x1] ) )/(nX*nY)

    # --------------
    # Run the SDP
    # --------------
    
    obj = cp.Maximize(QRAC)
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
    
    output += [QRAC.value]
    
    Mout = {}
    for y in range(nY):
        Mout[y] = {}
        for b in range(nB):
            Mout[y][b] = M[y][b].value
            
    output += [ Mout ]
    
    return output


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

def Shannon_Entropy(nX,nY,nB,m,t,w,monomials,gamma_matrix_els,xstar,ystar,W,energy):
    
    """
    Compute the Shannon entropy using a SDP relaxation with monomial list: {1,rho,M,phi}
    
    Inputs:
        nX:
            Number of state preparations
        nB:
            Number of measurement outcomes
        m:
            Gauss-Radau quadrature saturation limit
        t:
            Nodes from the Gauss-Radau quadrature
        w:
            Weigths from the Gauss-Radau quadrature
        pbxy:
            Full distribution of observable probabilities
        rhot:
            Target state
        eps:
            Distrust
        monomials:
            List of monomials used to build the SDP relaxation
        gamma_matrix_els:
            Set of identities and full matrix from the SDP relaxation after applying the relaxation rules
        xstar:
            Setting to extract randomness
        W:
            witness value (if any)
    """
    
    tau = [ w[i]/(t[i]*np.log(2.0)) for i in range(m) ]
    H_out = sum([ w[i]/(t[i]*np.log(2.0)) for i in range(m) ])
    
    [w_R,w_B,w_S] = monomials
    [G_new,map_table,S,list_of_eq_indices,Mexp] = gamma_matrix_els
    
    for i in range(m):
        
        print('\r',f'Running: {np.round(i/m*100,1)}%','\r',end="")
            
        #----------------------------------------------------#
        #                  CREATE VARIABLES                  #
        #----------------------------------------------------#
        
        z_var = {}
        h_var = {}
        for o in range(nB):    
            z_var[o] = cp.Variable(nonpos=True)
            h_var[o] = cp.Variable(nonneg=True)
    
        G_var_vec = {}
        for element in list_of_eq_indices:
            if element == map_table[-1][-1]:
                G_var_vec[element] = 0.0 # Zeros form orthogonal projectors
            else:
                G_var_vec[element] = cp.Variable(nonneg=True)
            
        zG_var_vec = {}
        hG_var_vec = {}
        for b in range(nB):
            zG_var_vec[b] = {}
            hG_var_vec[b] = {}
            for element in list_of_eq_indices:
                if element == map_table[-1][-1]:
                    zG_var_vec[b][element] = 0.0 # Zeros form orthogonal projectors
                    hG_var_vec[b][element] = 0.0 # Zeros form orthogonal projectors
                else:
                    zG_var_vec[b][element] = cp.Variable(nonpos=True)
                    hG_var_vec[b][element] = cp.Variable(nonneg=True)
      
        #--------------------------------------------------#
        #                  BUILD MATRICES                  #
        #--------------------------------------------------#
        
        G = {}
        lis = []
        for r in range(len(G_new)):
            lis += [[]]
            for c in range(len(G_new)):
                lis[r] += [G_var_vec[G_new[r][c]]]
        G = cp.bmat(lis)
            
        zG = {}
        hG = {}
        for b in range(nB):
            zlis = []
            hlis = []
            for r in range(len(G_new)):
                zlis += [[]]
                hlis += [[]]
                for c in range(len(G_new)):
                    zlis[r] += [zG_var_vec[b][G_new[r][c]]]
                    hlis[r] += [hG_var_vec[b][G_new[r][c]]]
            zG[b] = cp.bmat(zlis)
            hG[b] = cp.bmat(hlis)
                
        #------------------------------------------------------#
        #                  CREATE CONSTRAINTS                  #
        #------------------------------------------------------#
        
        ct = []
        
        
        # # Normalisation constraints --------------------------------------------------------------
        for y in range(nY):
            map_table_copy = map_table[:]
            
            identities = [ term[0] for term in map_table_copy]
            norm_cts = normalisation_contraints(w_B[y],identities)
            
            for gg in range(len(norm_cts)):
                the_elements = [fmap(map_table,norm_cts[gg][jj]) for jj in range(nB+1) ]
                an_element_is_not_in_the_list = False
                for hhh in range(len(the_elements)):
                    if the_elements[hhh] == 'ERROR: The value does not appear in the mapping rule':
                        an_element_is_not_in_the_list = True
                if an_element_is_not_in_the_list == False:
                    ct += [ sum([ G_var_vec[fmap(map_table,norm_cts[gg][jj])] for jj in range(nB) ]) == G_var_vec[fmap(map_table,norm_cts[gg][nB])] ]
                    for o in range(nB):
                        ct += [ sum([ zG_var_vec[o][fmap(map_table,norm_cts[gg][jj])] for jj in range(nB) ]) == zG_var_vec[o][fmap(map_table,norm_cts[gg][nB])] ]
                        ct += [ sum([ hG_var_vec[o][fmap(map_table,norm_cts[gg][jj])] for jj in range(nB) ]) == hG_var_vec[o][fmap(map_table,norm_cts[gg][nB])] ]
        # # ----------------------------------------------------------------------------------------
    
        tol = 1e-8
        # Positivity of tracial matrices and localising matrices
        for b in range(nB):
             
            Gamma = cp.bmat([[ G   ,zG[b]],
                             [zG[b],hG[b]] ])
            ct += [Gamma >> 0.0]
                
        # Some specific constraints in each corr matrix  -- G
    
        # Rank-1 projectors
        ct += [ G_var_vec[fmap(map_table,[w_R[x]])] == 1.0 for x in range(nX)]
        ct += [ G_var_vec[fmap(map_table,[w_S[k]])] == 1.0 for k in range(nK)]
        ct += [ G_var_vec[fmap(map_table,[w_B[y][b]])] == 1.0 for y in range(nY) for b in range(nB)]
        
        for o in range(nB):
            
            # Rank-1 projectors
            ct += [ zG_var_vec[o][fmap(map_table,[w_R[x]])] == z_var[o] for x in range(nX)]
            ct += [ zG_var_vec[o][fmap(map_table,[w_S[k]])] == z_var[o] for k in range(nK)]
            ct += [ zG_var_vec[o][fmap(map_table,[w_B[y][b]])] == z_var[o] for y in range(nY) for b in range(nB)]
            ct += [ zG_var_vec[o][fmap(map_table,[w_S[0],w_R[x]])] == z_var[o]*(1.0-energy) for x in range(nX) ]
    
            ct += [ hG_var_vec[o][fmap(map_table,[w_R[x]])] == h_var[o] for x in range(nX)]
            ct += [ hG_var_vec[o][fmap(map_table,[w_S[k]])] == h_var[o] for k in range(nK)]
            ct += [ hG_var_vec[o][fmap(map_table,[w_B[y][b]])] == h_var[o] for y in range(nY) for b in range(nB)]
            ct += [ hG_var_vec[o][fmap(map_table,[w_S[0],w_R[x]])] == h_var[o]*(1.0-energy) for x in range(nX) ]

        # Photon number avg constraint
        ct += [ G_var_vec[fmap(map_table,[w_R[x],w_S[0]])] == 1.0-energy for x in range(nX) ] 
        
        # Witness or full distribution
        ct += [ sum([ G_var_vec[fmap(map_table,[w_R[x],w_B[y][x]])]/float(nX) for x in range(nX)]) == W for y in range(nY)  ]
        
        ct += [ sum([ zG_var_vec[o][fmap(map_table,[w_R[x],w_B[y][x]])]/float(nX) for x in range(nX)]) == W*z_var[o] for y in range(nY) for o in range(nB)  ]
        ct += [ sum([ hG_var_vec[o][fmap(map_table,[w_R[x],w_B[y][x]])]/float(nX) for x in range(nX)]) == W*h_var[o] for y in range(nY) for o in range(nB)  ]
        
        # Shannon entropy
        H = 0.0
        for b in range(nB):
            H += w[i]/(t[i]*np.log(2.0)) * ( 2.0*zG_var_vec[b][fmap(map_table,[w_B[ystar][b],w_R[xstar]])] + 
                                      (1.0-t[i])*hG_var_vec[b][fmap(map_table,[w_B[ystar][b],w_R[xstar]])] + 
                                           t[i] *hG_var_vec[b][fmap(map_table,[w_R[xstar]])] ) 

        ct += [H >= -666.0] # To detect unbounded solutions
            
        #----------------------------------------------------------------#
        #                  RUN THE SDP and WRITE OUTPUT                  #
        #----------------------------------------------------------------#
    
        obj = cp.Minimize(H)
        prob = cp.Problem(obj,ct)
    
        output = []
    
        try:
            mosek_params = {
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
                }
            prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)
    
        except SolverError:
            something = 10
            
        if H.value != None:
            H_out += H.value
        else:
            H_out = None
            break
            
    return H_out

def prepare_hierarchy_Sh_SDI(nX,nY,nB,nK,some_second):
    
    # Track operators in the tracial matrix
    w_R = []
    w_B = [] # Bob
    w_S = []

    S_1 = [] # List of first order elements
    S_1_C = [] # List of first order elements (classical)
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
             for x in range(nX):
                 for y in range(nY):
                     S_high += [[w_B[y][b],w_R[x]]]
                 
        for x in range(nX):     
             for xx in range(nX):
                 S_high += [[w_R[xx],w_R[x]]]
                 
        for x in range(nX):     
             for k in range(nK):
                 S_high += [[w_R[x],w_S[k]]]
                 
        for x in range(nX):     
            for xx in range(nX):
                 for xxx in range(nX):
                     S_high += [[w_R[xxx],w_R[xx],w_R[x]]]
                 
    else:
         S_2 = S_1
         
         for x in range(nX):     
              for xx in range(nX):
                  for xxx in range(nX):
                      S_high += [[w_R[xxx],w_R[xx],w_R[x]]]
                      for xxxx in range(nX):
                          S_high += [[w_R[xxxx],w_R[xxx],w_R[xx],w_R[x]]]
                      
                  for k in range(nK):
                      S_high += [[w_S[k],w_R[xx],w_R[x]]]
                      S_high += [[w_R[xx],w_S[k],w_R[x]]]
                      
              for k in range(nK):
                  for kk in range(nK):
                      S_high += [[w_S[kk],w_R[x],w_S[k]]]
                      
         for b in range(nB):  
             for y in range(nY):
                 for x in range(nX):
                     for k in range(nK):
                         S_high += [[w_B[y][b],w_R[x],w_S[k]]]
                         S_high += [[w_B[y][b],w_S[k],w_R[x]]]
                         S_high += [[w_S[k],w_B[y][b],w_R[x]]]
                         
         for b in range(nB):  
             for y in range(nY):
                 for x in range(nX):
                     for xx in range(nX):
                         S_high += [[w_B[y][b],w_R[x],w_R[xx]]]
                         S_high += [[w_R[x],w_B[y][b],w_R[xx]]]
                         
                         
         for b in range(nB):
             for y in range(nY):
                 for bb in range(nB):
                     for yy in range(nY):
                         for x in range(nX):
                             S_high += [[w_B[y][b],w_R[x],w_B[yy][bb]]]
             
        
    # Set the operational rules within the SDP relaxation
    list_states = [] # operators that do not commute with anything (not important here)

    rank_1_projectors = []
    rank_1_projectors += w_R
    rank_1_projectors += [ w_B[y][b] for b in range(nB) for y in range(nY) ] 
    rank_1_projectors += w_S

    orthogonal_projectors = []
    orthogonal_projectors += [ [ w_B[y][b] for b in range(nB) ] for y in range(nY) ] 
    orthogonal_projectors += [w_S] 

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

dim = 2

xstar = 0
ystar = 0

N = 50 # number of datapoints

some_second = False

# energy = 0.2
# vec = np.linspace(psN(nX,energy)*0.93,psN(nX,energy),N)

energy = 0.005
vec = np.linspace(psN(nX,energy)*0.91,psN(nX,energy),N)
# vec = np.linspace(0.54,0.57,N)
Shannon_SDI_vec = [[],[]]

# Gauss-Radau quadrature weitghts (w) and nodes (t)
m_in = 5 # half of the quadrature limit ( m = m_in * 2 )
m = int(m_in*2) # quadrature limit
distribution = chaospy.Uniform(lower=1e-3, upper=1)
t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
t = t[0]

monomials, gamma_matrix_els = prepare_hierarchy_Sh_SDI(nX,nY,nB,nK,some_second)

for ii in range(N):
    
    Wobs = vec[ii]
    
    start = time.process_time()
    out_S = Shannon_Entropy(nX,nY,nB,m-1,t,w,monomials,gamma_matrix_els,xstar,ystar,Wobs,energy)
    end = time.process_time()
    
    Shannon_SDI_vec[0] += [vec[ii]]
    Shannon_SDI_vec[1] += [out_S]
        
    print(out_S, 'in',np.round(end-start,2),'seconds')
    
    
    
    
    
    
    
    
    
    





