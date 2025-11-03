#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 17:54:36 2025

@author: carles
"""

import numpy as np
import cvxpy as cp
from cvxpy import *
import time

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
id_2 = np.eye(2)

def embed(mat,D):
    
    """ Embed a d dimensional matrix "mat" into a D dimensional space """
    d = len(mat)
    if D >= d:
        
        M = np.zeros((D,D))
        M[:d,:d] = mat
        return M

    else:
        return ' d must be equal or smaller than D! ' 
  
def random_unitary_matrix(n):
    """Generates a random n x n unitary matrix"""
    XX = np.random.randn(n, n) #+ 1j*np.random.randn(n, n)  # Generate a random n x n complex matrix
    Q, R = np.linalg.qr(XX)  # Perform QR decomposition
    D = np.diagonal(R)  # Extract the diagonal elements of R
    P = D / np.abs(D)  # Compute the phase of the diagonal elements
    U = Q @ np.diag(P)  # Compute the unitary matrix
    return U

def partial_trace(rho, dims, system):
    """
    Compute the partial trace of a bipartite quantum state.

    Parameters:
    rho (numpy.ndarray): The density matrix of the bipartite system.
    dims (tuple): A tuple (dimA, dimB) representing the dimensions of the two subsystems.
    system (int): The subsystem to trace out (0 for subsystem A, 1 for subsystem B).

    Returns:
    numpy.ndarray: The reduced density matrix after tracing out the specified subsystem.
    """
    dimA, dimB = dims
    if system == 0:  # Trace out subsystem A
        reshaped_rho = rho.reshape(dimA, dimB, dimA, dimB)
        traced_out_rho = np.trace(reshaped_rho, axis1=0, axis2=2)
    elif system == 1:  # Trace out subsystem B
        reshaped_rho = rho.reshape(dimA, dimB, dimA, dimB)
        traced_out_rho = np.trace(reshaped_rho, axis1=1, axis2=3)
    else:
        raise ValueError("System must be 0 (for subsystem A) or 1 (for subsystem B).")

    return traced_out_rho


def generate_equidistant_states(nX,olap):
    
    """ Generates nX states in nX dimensions, all with the same overlap olap """
    
    rho = []
    
    state = []
    
    vec = np.zeros((nX,1))
    vec[0][0] = 1.0
    state += [vec]
    
    vec = np.zeros((nX,1))
    vec[0][0] = olap
    vec[1][0] = np.sqrt(1.0-vec[0][0]**2.0)
    state += [vec]

    for x in range(2,nX):
        
        A = np.array([ np.transpose(state[y])[0][:x] for y in range(0,x)])
        B = np.array([olap for y in range(0,x)])
                
        state_x = list(np.linalg.solve(A,B))
        state_x += [np.sqrt(1.0 - sum([state_x[u]**2.0 for u in range(len(state_x))]))]
        state_x = np.transpose(np.array([state_x]))

        final_state = np.zeros((nX,1))

        final_state[:len(state_x)] = state_x

        state += [final_state]
        
    for element in state:
        rho += [np.kron(element,np.transpose(element))]
    
    return rho

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Seesaw optimisation over channels and measurement (and fixed state)
def seesaw_EAPM_SD(dim,nX,nB,energy,E0_in):
    
    '''Maximum Sd for 2 channels (Alice0 and Alice1) sharing a bi-partite state (psi)'''
    '''and one measurement (Bob)'''
            
    ref = 0.0 #here store optimal Sd value per round of seesaw
    disp = 10.0 #displacement per run
    disp_avg = 10.0 #average of displacement
    eps = 1e-4 #averaged error allowed in seesaw
    tries = 20 #total number of runs per loop
    d = dim
    nC = nB
    
    #Initialise variables ------------------------------------
    ancilla = np.zeros((d**2,d**2))
    ancilla[0][0] = 1.0
    
    M = {}
    for b in range(nB):
        M[b] = cp.Variable((d**2,d**2),complex=False)
  
    
    R = {}
    U = random_unitary_matrix(d**2)
    olap = (nX*(1.0-energy)-1.0)/(nX-1)
    states = generate_equidistant_states(nX,olap)
    for x in range(nX):
        R[x] = U @ embed(states[x],d**2) @ np.transpose(np.conjugate(U))
 
    #---------------------------------------------------------
    
    var = 'M' #take Pi as the initial variable
    while disp_avg > eps:
        
        disp_avg = 0.0
#-------#tries ---------------------------------------------------
        for j in range(tries):
            
            ct = []

            if var == 'M':
                
                ct += [ M[b] >> 0.0 for b in range(nB) for y in range(nY) ]
                ct += [ M[b] == M[b].H for b in range(nB) for y in range(nY) ]
                ct += [ sum([M[b] for b in range(nB)]) == np.identity(d**2) for y in range(nY) ]
                                
            elif var == 'R':
                                
                ct += [ R[x] >> 0.0 for x in range(nX) ]
                ct += [ R[x] == R[x].H for x in range(nX) ]
                ct += [ cp.trace(R[x]) == 1.0 for x in range(nX) ]
                ct += [ cp.partial_trace(R[x],(d,d),0) == cp.partial_trace(R[xx],(d,d),0) for x in range(nX) for xx in range(nX) ]
                
                vacc = np.zeros((d,d))
                vacc[0][0] = 1.0
                vacc_x_id = np.kron(vacc,np.identity(d))
                
                ct += [ (cp.trace( R[x] @ vacc_x_id )) >= 1.0-energy for x in range(nX) ]
                                
            pbx = {}
            for b in range(nB):
                pbx[b] = {}
                for x in range(nX):
                    pbx[b][x] = ( cp.trace( R[x] @ M[b] ) )
                    
            E0 = (pbx[0][0] - pbx[1][0])
            E1 = (pbx[0][1] - pbx[1][1])
            
            ct += [ E0 == E0_in ]
                
            obj = cp.Maximize(E1)
            prob = cp.Problem(obj,ct)
            
            try:
                mosek_params = {"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1}
                prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

            except SolverError:
                something = 10
            
            if E1.value != None:

                disp = np.abs(E1.value - ref)
                disp_avg += disp/float(tries)
                
                #rotate variables --------------------------------
                if var == 'M':
                    var = 'R'

                    M_new = {}
                    for b in range(nB):
                        M_new[b] = M[b].value
                    M_old = M_new
                    M = M_new

                    R_old = R
                    R_new = {}
                    for x in range(nX):
                        R_new[x] = cp.Variable((d**2,d**2),complex=False)
                    R = R_new
                    
                elif var == 'R':
                    var = 'M'
                    
                    R_new = {}
                    for x in range(nX):
                        R_new[x] = R[x].value
                    R_old = R_new
                    R = R_new
                    
                    M_old = M
                    M_new = {}
                    for b in range(nB):
                        M_new[b] = cp.Variable((d**2,d**2),complex=False)
                    M = M_new

                #-------------------------------------------------

                ref = E1.value
                string = f'E1: {E1.value}, Displacement: {disp}, Avg_disp: {disp_avg}, Try:{j}, Var:{var}'
                print('\r'+string+'\r',end='')

            else: #if the solver does not find a solution, start again

                string = f'Try:{j}, Value not found. Start again....\r'
                print(string, end='')
                output = seesaw_EAPM_SD(dim,nX,nB,energy,E0_in)
                
                return output
            
#-------#tries ---------------------------------------------------

    #write the output --------------------------------
    output = []
    
    output += [ E1.value ]
    
    Rout = {}
    for x in range(nX):
        Rout[x] = R_old[x]
    
    output += [Rout]
    
    Mout = {}
    for b in range(nB):
        Mout[b] = M_old[b]
    
    output += [Mout]
    
    #-------------------------------------------------
            
    return output

#------------------------------------------------------------------------------------------

nX = 2
nB = 2

dim = 2

N = 100 # number of datapoints

energy = 0.2#vec[ii]
gamma = 1.0 - 2.0*(1.0-energy)

vec = np.linspace(-0.999,-(1.0-2.0*gamma**2),N)
ps_seesaw_vec_E0E1 = [[],[]]
ref = -10.0
for ii in range(N):
    
    E0_in = -0.8#vec[ii]
    
    start = time.process_time()
    Nattempts = 10

    for h in range(Nattempts):
        out_Q = seesaw_EAPM_SD(dim,nX,nB,energy,E0_in)
        if out_Q[0] >= ref:
            ref = out_Q[0]
 
    out_Q = ref
    end = time.process_time()
   
    ps_seesaw_vec_E0E1[0] += [vec[ii]]
    ps_seesaw_vec_E0E1[1] += [out_Q]
    
    print(out_Q, 'in',np.round(end-start,2),'seconds')














