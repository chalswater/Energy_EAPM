import numpy as np
from scipy.optimize import minimize_scalar


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#                                  FUNCTIONS                                   #
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

# Unentangled qubit case

def W_PM(omega):
    if omega < 0 or omega > 1:
        raise ValueError("omega must be in the range [0, 1]")
    value = 0.5 * (np.sqrt(1 - omega) + np.sqrt(omega))**2
    return value


# Qubit-qubit case

def W_of_r_qubit(r, omega):
    z = 1 - omega

    # Avoid numerical errors leading to small negative sqrt arguments
    sqrt_term = (5 * r**2 
                 - 4 * r * omega 
                 + 8 * np.sqrt(z) * np.sqrt(omega - r) * np.sqrt((z - r) * (r + omega))
                 + 8 * z * omega)

    if sqrt_term < 0:
        return -np.inf  # outside physical domain

    l = 0.5 * (r + np.sqrt(sqrt_term))
    W = 0.5 * (1 + l)
    return W

def W_qubit(omega):
    # Search over the interval 0 <= r <= omega
    result = minimize_scalar(lambda r: -W_of_r_qubit(r, omega), 
                             bounds=(0, omega), 
                             method='bounded')
    
    if result.success:
        r_opt = result.x
        W_max = W_of_r_qubit(r_opt, omega)
        return r_opt, W_max
    else:
        raise RuntimeError("Optimization failed")


# Qutrit-qutrit case


def W_of_r_qutrit(r, omega):
    # a depends on r and omega
    a = 1 - r - omega
    
    # Compute the term inside the square root
    inner_sqrt = 2 * a * np.sqrt(omega**2 - r**2) + r**2 + 2 * omega * a
    
    # Ensure the argument is real
    if inner_sqrt < 0:
        return -np.inf  # invalid (complex) region

    l = r + np.sqrt(inner_sqrt)
    W = 0.5 * (1 + l)
    return W

def W_qutrit(omega):
    # Search over 0 <= r <= omega
    result = minimize_scalar(lambda r: -W_of_r_qutrit(r, omega),
                             bounds=(0, omega),
                             method='bounded')
    
    if result.success:
        r_opt = result.x
        W_max = W_of_r_qutrit(r_opt, omega)
        return r_opt, W_max
    else:
        raise RuntimeError("Optimization failed")

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#                                  MAIN CODE                                   #
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#


N = 1000

PM_W_vec = [[],[]]
EAPM_W_vec_qubit = [[],[]]
EAPM_W_vec_qutrit = [[],[]]

vec = np.linspace(0.0,1.0/2.0,N)

for i in range(N):
    
    omega = 0.2#vec[i]
    
    W_PM_val = W_PM(omega)
    PM_W_vec[0] += [ omega ]
    PM_W_vec[1] += [ W_PM_val ]
    
    r_opt_qubit, W_max_qubit = W_qubit(omega)
    EAPM_W_vec_qubit[0] += [ omega ]
    EAPM_W_vec_qubit[1] += [ W_max_qubit ]
    
    r_opt_qutrit, W_max_qutrit = W_qutrit(omega)
    EAPM_W_vec_qutrit[0] += [ omega ]
    EAPM_W_vec_qutrit[1] += [ W_max_qutrit ]
    
    print(W_PM_val,W_max_qubit,W_max_qutrit)