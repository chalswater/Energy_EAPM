#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 14:37:04 2025

@author: carles
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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


# fig, ax = plt.subplots(1,2,figsize=(5, 2))
fig, ax = plt.subplots(1,2,figsize=(6, 2))

ax[0].set_xlim(0.839,0.901)
# ax[0].set_ylim(-0.005,0.155)

ax[0].set_ylim(-0.005,0.35)

ax[1].set_xlim(0.518,0.572)
# ax[1].set_ylim(-0.03,0.85)

ax[1].set_ylim(-0.03,1.0)

# ax[0].set_xticks([0.84,0.86,0.88,0.9])
# ax[0].set_yticks([0.0,0.05,0.1,0.15])

ax[1].set_xticks([0.52,0.53,0.54,0.55,0.56,0.57])
ax[1].set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])

# ax[0].set_title(r'$\omega = 0.2$',pad=-5)
# ax[1].set_title(r'$\omega = 0.05$')

fig.text(0.11,0.75,r'$\omega = 0.2$')
fig.text(0.61,0.75,r'$\omega = 0.005$')

ax[0].set_xlabel(r'$\mathcal{W}_{2}$',labelpad=-1)
ax[0].set_ylabel(r'Randomness',labelpad=1.0)

ax[1].set_xlabel(r'$\mathcal{W}_{2}$',labelpad=-1)
ylabel = ax[1].set_ylabel(r'Randomness',labelpad=1.0)


HminQ, = ax[0].plot(QRNG_seesaw_d3_w_02[0],QRNG_seesaw_d3_w_02[1],c='purple',marker='s',ms=3,label=' ',alpha=0.3,ls='')
HminC, = ax[0].plot(QRNG_classical_d3_w_02[0],QRNG_classical_d3_w_02[1],c='green',marker='',label=' ',alpha=0.3,ls='--')

ShannonQ, = ax[0].plot(Shannon_upper_w_02[0],Shannon_upper_w_02[1],c='purple',marker='.',label=' ',ls='')
ShannonC, = ax[0].plot(Shannon_classical_d3_w_02[0],Shannon_classical_d3_w_02[1],c='green',marker='',label=' ')


ax[1].plot(QRNG_seesaw_d3_w_0005[0],QRNG_seesaw_d3_w_0005[1],c='purple',marker='s',ms=3,alpha=0.3,ls='')
ax[1].plot(QRNG_classical_d3_w_0005[0],QRNG_classical_d3_w_0005[1],c='green',marker='',alpha=0.3,ls='--')

ax[1].plot(Shannon_upper_w_0005[0],Shannon_upper_w_0005[1],c='purple',marker='.',ls='')
ax[1].plot(Shannon_classical_d3_w_0005[0],Shannon_classical_d3_w_0005[1],c='green',marker='')

ax[0].set_title('High-energy regime',size=9)
ax[1].set_title('Low-energy regime',size=9)

# ax[0].plot(result_vec[0],result_vec[1])
# ax[1].plot(Shannon_SDI_vec[0],Shannon_SDI_vec[1])

height = -0.06
width = -0.0

quantum = fig.text(1.18+width,0.68+height,'Quantum')
attacks = fig.text(1.192+width,0.6+height,'attacks')

classical = fig.text(1.18+width,0.44+height,'Classical')
side_info = fig.text(1.182+width,0.36+height,'side info.')

Hmin = fig.text(1.+width,0.8+height,'$H_{min}$')
# H = fig.text(1.085+width,0.8+height,'Shannon')
H = fig.text(1.119+width,0.8+height,'VN')

legQ = fig.legend(handles=[HminQ,ShannonQ],bbox_to_anchor=(1.2+width, 0.73+height),fontsize=9,facecolor='white', framealpha=1,ncols=2,frameon=False)
legC = fig.legend(handles=[HminC,ShannonC],bbox_to_anchor=(1.2+width, 0.5+height),fontsize=9,facecolor='white', framealpha=1,ncols=2,frameon=False)

fig.add_artist(legQ)
fig.add_artist(legC)
fig.add_artist(ylabel)

fig.add_artist(quantum)
fig.add_artist(attacks)

fig.add_artist(classical)
fig.add_artist(side_info)

fig.subplots_adjust(hspace=0.28)

# ax[0].plot(result_vec[0],result_vec[1],c='red',marker='.',ms=6,label=' ',alpha=1,ls='')

fig.tight_layout()

# fig.savefig('QRNG_Q_Cl_side_info.pdf',bbox_inches='tight')






















