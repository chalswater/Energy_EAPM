import numpy as np
from scipy.optimize import minimize

def psN(n,energy):

    alpha = 1.0-energy

    if energy <= 1.0-1.0/n:

        return (np.sqrt((n-1)*(energy))+np.sqrt(1-energy))**2/n
    
    else:
        
        return 1.0

n = 2 # Number of messages
nL = 4 # Cardinality of lambda
d = 3 # Dimension
eps = 1e-9 # Error tolerance

N = 20

dim = d*d
num_rho = n * n
chol_size = dim * dim  # params for each PSD matrix by chol factor

# vacc matrix (only first element 1)
vacc = np.zeros((d, d))
vacc[0, 0] = 1.0

w = 0.2
vec = np.linspace(psN(nX,w)*0.93,psN(nX,w),N)

# w = 0.005
# vec = np.linspace(psN(nX,w)*0.91,psN(nX,w),N)

result_vec = [[],[]]

for ii in range(N):
    
    S = vec[ii]
    
    def random_unitary_matrix(n):
        """Generates a random n x n unitary matrix"""
        XX = np.random.randn(n, n) #+ 1j*np.random.randn(n, n)  # Generate a random n x n complex matrix
        Q, R = np.linalg.qr(XX)  # Perform QR decomposition
        D = np.diagonal(R)  # Extract the diagonal elements of R
        P = D / np.abs(D)  # Compute the phase of the diagonal elements
        U = Q @ np.diag(P)  # Compute the unitary matrix
        return U
    
    def vec_to_psd(chol_vec,dimension):
        L = chol_vec.reshape(dimension, dimension)
        A = L @ L.T
        return A
    
    def normalize_density(rho):
        return rho / np.trace(rho)
    
    def partial_trace_A(rho, d):
        return np.trace(rho.reshape(d, d, d, d), axis1=0, axis2=2)
    
    def partial_trace_B(rho, d):
        return np.trace(rho.reshape(d, d, d, d), axis1=1, axis2=3)
    
    def softmax(x):
        ex = np.exp(x - np.max(x))
        return ex / ex.sum()
    
    def clip_eig(M):
        e, v = np.linalg.eigh(M)
        e_clipped = np.clip(e, 0, None)
        return v @ np.diag(e_clipped) @ v.T
    
    # Unpack variables: rho_x^l, M_b, q, and sigma_B
    def unpack_vars(x):
        offset = 0
        rhos = []
        for xi in range(n):
            rhos += [[]]
            for l in range(n):
                chol_vec = x[offset:offset + dim * dim]
                offset += dim * dim
                rho = normalize_density(vec_to_psd(chol_vec,dim))
                rhos[xi] += [ rho ]
        
        Ms = []
        for _ in range(n-1):
            chol_vec = x[offset:offset + dim * dim]
            offset += dim * dim
            M = clip_eig(vec_to_psd(chol_vec,dim))
            Ms.append(M)
        Ms.append(np.identity(dim)- sum([ Ms[b] for b in range(n-1) ]))
        Ms = np.array(Ms)
        
        q_raw = x[offset:offset + n]
        offset += n
        q = softmax(q_raw)
        
        # sigma_B variable: d x d PSD matrix normalized
        sigmaB_chol = x[offset:offset + d * d]
        sigma_B = normalize_density(vec_to_psd(sigmaB_chol,d))
        
        return rhos, Ms, q, sigma_B
    
    def objective(x):
        rhos, Ms, q, sigma_B = unpack_vars(x)
    
        # Normalize POVM elements to sum to identity approx
        Msum = np.sum(Ms, axis=0)
        Msum = clip_eig(Msum)
        # Inverse sqrt for normalization
        U = np.linalg.cholesky(Msum + eps * np.eye(dim))
        U_inv = np.linalg.inv(U)
        for i in range(n):
            Ms[i] = U_inv.T @ Ms[i] @ U_inv
            Ms[i] = clip_eig(Ms[i])
    
        fixed_x = 0
        total = 0
        for l in range(n):
            for b in range(n):
                idx = fixed_x * n + l
                p = np.trace(rhos[fixed_x][l] @ Ms[b])
                if p <= 1e-6:
                    p = 1e-6
                total += q[l] * p * np.log2(p)
        return -total
    
    # Constraint: tr_A(rho_x^l) == sigma_B for all x,l
    def constraint_nonsignalling(x):
        rhos, Ms, q, sigma_B = unpack_vars(x)
        output = []
        val = 0
    
        for xi in range(n):
            for l in range(n):
                trA = partial_trace_A(rhos[xi][l], d)
                diff = trA - sigma_B
                val += np.linalg.norm(diff, 'fro')**2
                output += [val]
    
        return np.array(output).flatten()
    
    # Constraint: sum_l q(l) * (1/n) sum_x tr(tr_B(rho_x^l) @ vacc) >= 1 - w
    def constraint_energy(x):
        rhos, Ms, q, sigma_B = unpack_vars(x)
        output = []
        for xi in range(n):
            suma = 0
            for l in range(n):
                trB = partial_trace_B(rhos[xi][l], d)
                suma += q[l] * np.trace(trB @ vacc)
            output += [ suma - (1-w) ]
        return np.array(output).flatten()
        # return val - (1 - w)  # inequality >= 0
    
    # Constraint: average sum_x sum_l q(l) p(x|x,l)/n >= S
    def constraint_state_disc(x):
        rhos, Ms, q, sigma_B = unpack_vars(x)
        s = 0
        for l in range(n):
            for xi in range(n):
                p_val = q[l] * np.trace(rhos[xi][l] @ Ms[xi])/n  # p(b=x_i|x_i,l)
                s += p_val
        return s - S  # inequality >= 0
    
    # Constraint: trace of sigma_B == 1 (density matrix)
    def constraint_sigmaB_trace(x):
        _, _, _, sigma_B = unpack_vars(x)
        diff = np.trace(sigma_B) - 1.0 
        # val = np.linalg.norm(diff, 'fro')**2
        return diff
    
    def normalisaton_constraint(x):
        _, Ms, _, _ = unpack_vars(x)
        normal = Ms[0] + Ms[1] - np.identity(4)
        return normal.flatten()
    
    def psd_constraint(x):
        rhos, Ms, q, sigma_B = unpack_vars(x)
        eigenvalues = []
        for l in range(n):
            for xi in range(n):
                eigs, _ = np.linalg.eigh(rhos[xi][l])
                eigenvalues += list(eigs)
            
        for b in range(n):
            eigs, _ = np.linalg.eigh(Ms[b])
            eigenvalues += list(eigs)
            
        eigs, _ = np.linalg.eigh(sigma_B)
        eigenvalues += list(eigs)
        
        return np.array(eigenvalues).flatten()
        
    
    # Initial guess:
    np.random.seed(np.random.randint(10))
    
    rho_ini = []
    for l in range(nL):
        rho_ini += [[]]
        for xi in range(n):
            UU = random_unitary_matrix(dim)
            zeros = np.zeros((dim,dim))
            zeros[0][0] = 1.0
            rho_ini[l] += [ UU @ zeros @ UU.T ]
    x0_rho = np.array(rho_ini).flatten()
    
    
    UU = random_unitary_matrix(dim)
    M_ini = []
    for b in range(n):
        zeros = np.zeros((dim,dim))
        if b == 0:
            zeros[0][0] = 1.0
            zeros[1][1] = 1.0
        elif b == 1:
            zeros[2][2] = 1.0
            zeros[3][3] = 1.0
        M_ini += [ UU @ zeros @ UU.T ]

    x0_M = np.array(M_ini).flatten()
    
    x0_q = np.random.randint(0,100,n)
    x0_q = x0_q / np.sum(x0_q)
    
    U = random_unitary_matrix(d)        
    
    sigma_ini = np.zeros((d,d))
    sigma_ini[0][0] = 1.0
    sigma_ini = U @ sigma_ini @ U.T
    x0_sigmaB = np.array(sigma_ini).flatten()
    
    x0 = np.concatenate([x0_rho, x0_M, x0_q, x0_sigmaB])
    
    constraints = [
        {'type': 'eq', 'fun': constraint_nonsignalling},
        {'type': 'ineq', 'fun': constraint_energy},
        {'type': 'ineq', 'fun': constraint_state_disc},
        {'type': 'ineq', 'fun': psd_constraint},
        {'type': 'ineq', 'fun': objective}
    ]
    
    method_in = 'SLSQP'
    
    res = minimize(objective, x0, method=method_in, constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True})
    
    print("Success:", res.success)
    print("Optimal value (negative entropy):", res.fun)
    
    rhos_opt, Ms_opt, q_opt, sigmaB_opt = unpack_vars(res.x)
    
    result_vec[0] += [ S ]
    result_vec[1] += [ res.fun ]
