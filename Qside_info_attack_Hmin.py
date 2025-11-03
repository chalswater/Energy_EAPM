#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:35:16 2025

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

def psN(n,energy):

    alpha = 1.0-energy

    if energy <= 1.0-1.0/n:

        return (np.sqrt((n-1)*(energy))+np.sqrt(1-energy))**2/n
    
    else:
        
        return 1.0
    

def psQ(Q,energy):
    
    olap = 1.0 - 2.0*energy
    
    if Q <= olap:
        WQ = 0.5*np.sqrt((1.0-olap**2)*(1.0-2*Q/(1.0+olap)))
        return 0.5*(1.0 + 2.0*WQ - Q)
    else:
        WQ = 0.5*(1.0-Q)
        return 0.5*(1.0 + 2.0*WQ - Q)

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


def seesaw_EAPM_SD_find_state(dim,nX,nB,energy):
    
    '''Maximum Sd for 2 channels (Alice0 and Alice1) sharing a bi-partite state (psi)'''
    '''and one measurement (Bob)'''
            
    w = energy
    ref = 0.0 #here store optimal Sd value per round of seesaw
    disp = 10.0 #displacement per run
    disp_avg = 10.0 #average of displacement
    eps = 1e-4 #averaged error allowed in seesaw
    tries = 20 #total number of runs per loop
    d = dim
    
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
    
    # Solution to qutrit states
    
    psi = []
    
    hh = np.sqrt(1.0-2.0*w)
    p0 = ( 1.0 - w - hh )/(1.0-hh)
    
    psi0 = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    psi1 = np.array([0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    
    r = 0.0#1-w/(1-np.sqrt(1-2*w))
    
    xx = np.sqrt((w+r)/(2*(1-r)))
    yy = np.sqrt((w-r)/(2*(1-r)))
    
    varphi0_ket = np.array([0.0 , 0.0,  np.sqrt(1.0-w/(1-r)), -xx, yy, 0.0, 0.0, 0.0, 0.0])
    varphi1_ket = np.array([0.0 , 0.0,  np.sqrt(1.0-w/(1-r)), yy, -xx, 0.0, 0.0, 0.0, 0.0])
    
    varphi0 = np.outer(varphi0_ket,varphi0_ket)
    varphi1 = np.outer(varphi1_ket,varphi1_ket)
    
    psi += [ p0 * np.outer(psi0,psi0) + (1.0-p0) * varphi0]
    psi += [ p0 * np.outer(psi1,psi1) + (1.0-p0) * varphi1]
    
    R[0] = psi[0]
    R[1] = psi[1]
    
    #---------------------------------------------------------
    
    var = 'M' #take Pi as the initial variable
    while disp_avg > eps:
        
        disp_avg = 0.0
#-------#tries ---------------------------------------------------
        for j in range(tries):
            
            ct = []

            if var == 'M':
                
                ct += [ M[b] >> 0.0 for b in range(nB) ]
                ct += [ M[b] == M[b].H for b in range(nB) ]
                ct += [ sum([M[b] for b in range(nB)]) == np.identity(d**2) ]
                
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
                
            ps = sum([ pbx[x][x]/nX for x in range(nX) ])
            
            ct += [ ps <= 666.0 ]
                
            obj = cp.Maximize(ps)
            prob = cp.Problem(obj,ct)
            
            try:
                mosek_params = {"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1}
                prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

            except SolverError:
                something = 10
            
            if ps.value != None:

                disp = np.abs(ps.value - ref)
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

                ref = ps.value
                string = f'Ps: {ps.value}, Displacement: {disp}, Avg_disp: {disp_avg}, Try:{j}, Var:{var}'
                print('\r'+string+'\r',end='')

            else: #if the solver does not find a solution, start again

                string = f'Try:{j}, Value not found. Start again....\r'
                print(string, end='')
                print(var)
                output = seesaw_EAPM_SD(dim,nX,nB,energy)
                
                return output
            
#-------#tries ---------------------------------------------------

    #write the output --------------------------------
    output = []
    
    output += [ ps.value ]
    
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

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#Seesaw optimisation over channels and measurement (and fixed state)
def seesaw_EAPM_QRNG(dim,nX,nB,energy,Wobs,R_ini,M_ini):
    
    '''Maximum Sd for 2 channels (Alice0 and Alice1) sharing a bi-partite state (psi)'''
    '''and one measurement (Bob)'''
            
    w = energy
    ref = 0.0 #here store optimal Sd value per round of seesaw
    disp = 10.0 #displacement per run
    disp_avg = 10.0 #average of displacement
    eps = 1e-4 #averaged error allowed in seesaw
    tries = 10 #total number of runs per loop
    d = dim
    nL = nB
    
    #Initialise variables ------------------------------------
    ancilla = np.zeros((d**2,d**2))
    ancilla[0][0] = 1.0
    
    U = random_unitary_matrix(d**2)
    
    M = {}
    for b in range(nB):
        # M[b] = cp.Variable((d**2,d**2),PSD=True)
        M[b] = M_ini[b] 
    
    q = np.random.randint(1, 100, size=nL)

    q = q / np.sum(q)
    
    R = {}
    for l in range(nL):
        R[l] = {}
        for x in range(nX):
            R[l][x] = cp.Variable((d**2,d**2),PSD=True)
            # R[l][x] = q[l]* R_ini[x]

    #---------------------------------------------------------
    
    var = 'R' #take Pi as the initial variable
    while disp_avg > eps: 
        
        disp_avg = 0.0
#-------#tries ---------------------------------------------------
        for j in range(tries):
            
            ct = []

            if var == 'M':

                ct += [ sum([M[b] for b in range(nB)]) == np.identity(d**2) ]
               
            elif var == 'R':
                
                q = cp.Variable(nL,nonneg=True)
                ct += [sum([q[l] for l in range(nL)]) == 1.0]

                ct += [ cp.trace(R[l][x]) == q[l] for x in range(nX) for l in range(nL) ]
                ct += [ cp.partial_trace(R[l][x],(d,d),0) == cp.partial_trace(R[l][xx],(d,d),0) for x in range(nX) for xx in range(nX) for l in range(nL) ]
        
                vacc = np.zeros((d,d))
                vacc[0][0] = 1.0
                vacc_x_id = np.kron(vacc,np.identity(d))

                ct += [ sum([ (cp.trace( R[l][x] @ vacc_x_id )) for l in range(nL) ]) >= 1.0-energy for x in range(nX) ]
                 
            pbx = {}
            for b in range(nB):
                pbx[b] = {}
                for x in range(nX):
                    pbx[b][x] = sum([ cp.trace( R[l][x] @ M[b] ) for l in range(nL) ])
                        
            ct += [ (sum([ pbx[x][x]/nX for x in range(nX) ])) >= Wobs ]
            
            ps = (sum([ pbx[x][x]/nX for x in range(nX) ]))
                        
            pg = sum([ (cp.trace( R[l][0] @ M[l] )) for l in range(nL) ]) #- ps
            
            obj = cp.Maximize(pg)
            prob = cp.Problem(obj,ct)
            
            try:
                mosek_params = {"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1}
                prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

            except SolverError:
                something = 10
            
            if pg.value != None:

                disp = np.abs(pg.value - ref)
                disp_avg += disp/float(tries)
                
                if pg.value >= 0.999:
                    disp_avg = 0.0
                
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
                    for l in range(nL):
                        R_new[l] = {}
                        for x in range(nX):
                            R_new[l][x] = cp.Variable((d**2,d**2),PSD=True)
                    R = R_new
                    
                elif var == 'R':
                    var = 'M'
                    
                    R_new = {}
                    for l in range(nL):
                        R_new[l] = {}
                        for x in range(nX):
                            R_new[l][x] = R[l][x].value
                    R_old = R_new
                    R = R_new
                    
                    M_old = M
                    M_new = {}
                    for b in range(nB):
                        M_new[b] = cp.Variable((d**2,d**2),PSD=True)
                    M = M_new

                #-------------------------------------------------

                ref = pg.value
                string = f'Pg: {pg.value}, Displacement: {disp}, Avg_disp: {disp_avg}, Try:{j}, Var:{var}'
                print('\r'+string+'\r',end='')
                # print(var)
            else: #if the solver does not find a solution, start again

                string = f'Try:{j}, Value not found. Start again....\r'
                print(string, end='')
                print(var)
                output = seesaw_EAPM_QRNG(dim,nX,nB,energy,Wobs,R_ini,M_ini)
                
                return output
            
#-------#tries ---------------------------------------------------

    #write the output --------------------------------
    output = []
    
    output += [ -np.log2(pg.value) ]
    
    #-------------------------------------------------
            
    return output

#------------------------------------------------------------------------------------------

nX = 2
nB = 2

xstar = 0
ystar = 0

N = 20 # number of datapoints

energy = 0.005

# Find initial states

vec = np.linspace(psN(nX,energy)*0.91,psN(nX,energy),N)

ps_seesaw_vec = [[],[]]

for ii in range(N):
        
    Wobs = vec[ii]
    
    dim = 3
    
    print('Dimension:',dim)
    
    print('Observed:',Wobs,'Max',psN(nX,energy))
    
    start = time.process_time()
    Nattempts = 1
    ref = 10.0
    for h in range(Nattempts):
        
        begin = seesaw_EAPM_SD_find_state(dim,nX,nB,energy)
        R_ini = begin[1]
        M_ini = begin[2]
        
        out_Q = seesaw_EAPM_QRNG(dim,nX,nB,energy,Wobs,R_ini,M_ini)
        
        if out_Q[0] <= ref:
            ref = out_Q[0]
            
    end = time.process_time()

    ps_seesaw_vec[0] += [vec[ii]]
    ps_seesaw_vec[1] += [ ref ]
    
    print(out_Q, 'in',np.round(end-start,2),'seconds')














