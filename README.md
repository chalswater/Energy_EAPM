# Energy-restricted EAPM
The codes listed in this repository are used to reproduce the results reported in the paper titled: "_The role of entanglement in energy-restricted communication and randomness generation_": https://arxiv.org/abs/2510.27473

The codes are structured as follows:

1. **Probabilistic bit-tramission part**: These codes are dedicated to compute the maximum success probability in bit transmission. These are:
   
   a) `analytical_W_bounds.py`: The analytical form of the optimal states is used to compute the maximum success probability with energy-restriced entanglement assistance.
   
   b) `E0_E1_seesaw.py`: Seesaw semidefinite program to compute the bounds on the (E0,E1) correlator space.

2. **Quantum random number generation**: These codes are used to compute bounds on the certifiable randomness assuming classical side information, and attacks from quantum eavesdroppers.
   
   a) `Cside_info_bound_Hmin.py`: Bound on the certifiable randomness assuming classical side information quantified through the min-entropy.
   
   b) `Cside_info_bound_Shannon.py`: Bound on the certifiable randomness assuming classical side information quantified through the Shannon entropy.
   
   c) `Qside_info_attack_Hmin.py`: Min-entropy computed assuming an attack with quantum side information.
   
   d) `Qside_info_attack_Hmin.py`: Von Neumann entropy computed an attack assuming quantum side information.

3. **Display results in plots using these codes**:
   
   a) `plotting_success_probs.py`: Figure 2 from the paper.
   
   b) `plotting_correlation_Em_Ep.py`: Figure 3 from the paper.
   
   c) `plot_QRNG_cases.py`: Figure 4 from the paper.


