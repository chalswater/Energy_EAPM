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
    
fig, ax = plt.subplots(figsize=(4, 3))

c0 = 'tomato'
c1 = 'darkgreen'
c2 = 'mediumblue'

plt.ylabel(r'$\mathcal{W}_{2}$')
plt.xlabel(r'$\omega$')

# plt.grid()

ax.set_ylim(0.495,1.01)
ax.set_xlim(-0.005,0.505)

# ax.axvline(np.sqrt(2)-1)

ax.plot(EAPM_W_vec_qutrit[0],EAPM_W_vec_qutrit[1],c=c0,marker='',label='EAPM $d=3$',lw=1.2)
ax.plot(EAPM_W_vec_qubit[0],EAPM_W_vec_qubit[1],c=c1,marker='',label='EAPM $d=2$',lw=1.2)
ax.plot(PM_W_vec[0],PM_W_vec[1],c=c2,label='PM',ls='-',lw=1.2)

# Seesaw values
# ax.plot(EAPM_SD_d4_seesaw[0],EAPM_SD_d4_seesaw[1],c='red',marker='.',ls='')
# ax.plot(EAPM_SD_d2_seesaw[0],EAPM_SD_d2_seesaw[1],c='green',marker='.',ls='')

# Inset plot
inset_ax = inset_axes(ax, width="50%", height="50%", bbox_to_anchor=(0.4, -0.3, 1.0, 1.0),
                      bbox_transform=ax.transAxes, loc='upper left')

inset_ax.set_ylim(0.9,0.97)
inset_ax.set_xlim(0.2,0.3)

# Seesaw values inside the inset plot
# inset_ax.plot(EAPM_SD_d2_seesaw[0],EAPM_SD_d2_seesaw[1],c='green',marker='.',ls='')
# inset_ax.plot(EAPM_SD_d4_seesaw[0],EAPM_SD_d4_seesaw[1],c='red',marker='.',ls='',markersize=5)

inset_ax.plot(EAPM_W_vec_qutrit[0],EAPM_W_vec_qutrit[1],c=c0,marker='',lw=1.2)
inset_ax.plot(EAPM_W_vec_qubit[0],EAPM_W_vec_qubit[1],c=c1,marker='',lw=1.2)
inset_ax.plot(PM_W_vec[0],PM_W_vec[1],c=c2,lw=1.2)

# inset_ax.grid()

# Draw lines from inset to zoom region on main plot
mark_inset(ax, inset_ax, loc1=3, loc2=1, fc="none", ec="0.5")

leg = fig.legend(bbox_to_anchor=(0.49, 0.97),fontsize=9,facecolor='white', framealpha=1,ncols=1,frameon=False)

fig.add_artist(leg)

fig.tight_layout()

# fig.savefig('PM_EAPM_SD_noseesaw.pdf')




















