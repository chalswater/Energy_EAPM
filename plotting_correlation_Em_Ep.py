#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:29:27 2025

@author: carles
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

chosen_omega = 0.2

def PM_ellipse(E0,w,direction):
    
    gamma = 1.0 - 2.0*(1.0-w)
    
    if direction == 'up':
        if E0 <= -(1.0-2.0*gamma**2):
            return -E0*(1.0-2.0*gamma**2) - 2.0*gamma*np.sqrt(1.0-gamma**2)*np.sqrt(1.0-E0**2)
        else:
            return 1.0
    elif direction == 'down':
        if E0 >= (1.0-2.0*gamma**2):
            return -E0*(1.0-2.0*gamma**2) + 2.0*gamma*np.sqrt(1.0-gamma**2)*np.sqrt(1.0-E0**2)
        else:
            return -1.0

fig, ax = plt.subplots(figsize=(4, 3.6))

c0 = 'tomato'
c1 = 'darkgreen'
c2 = 'mediumblue'

# plt.grid()

ax.set_xlabel(r'$E_0$',size=13)
ax.set_ylabel(r'$E_1$',size=13)

ax.set_ylim(-1.05,1.05)
ax.set_xlim(-1.05,1.05)

# plt.ylim(0.25,1.0)
# plt.xlim(-1.0,-0.35)

vecc = np.linspace(-1.0,1.0,500)
E1_PM_vec_up = [ PM_ellipse(vecc[i],chosen_omega,'up') for i in range(500) ]
E1_PM_vec_down = [ PM_ellipse(vecc[i],chosen_omega,'down') for i in range(500) ]

ax.plot(E1_E0_vec_qubit[0],E1_E0_vec_qubit[1],c=c1,marker='')
ax.plot(E1_E0_vec_qubit[1],E1_E0_vec_qubit[0],c=c1,marker='')

ax.plot(E1_E0_vec_qutrit[0],E1_E0_vec_qutrit[1],c=c0,marker='')
ax.plot(E1_E0_vec_qutrit[1],E1_E0_vec_qutrit[0],c=c0,marker='')

ax.plot([-1.0] + list(E1_E0_vec_qutrit[0]) + [1.0], [-1.0] + list(E1_E0_vec_qutrit[1]) + [1.0],c=c0,marker='',markersize=5)
ax.plot([-1.0] + list(E1_E0_vec_qutrit[1]) + [1.0], [-1.0] + list(E1_E0_vec_qutrit[0]) + [1.0],c=c0,marker='',markersize=5)

vecc = [-1.0] + list(vecc)

E1_PM_vec_up = [-1.0] + E1_PM_vec_up
E1_PM_vec_down = E1_PM_vec_down + [1.0]

ax.plot(vecc,E1_PM_vec_up,c=c2,label='PM',lw=1.2)
ax.plot(vecc,E1_PM_vec_down,c=c2,lw=1.2)

ax.fill_between(vecc, vecc, E1_PM_vec_up, color=c2,edgecolor='none',alpha=0.3)
ax.fill_between(vecc, vecc, E1_PM_vec_down, color=c2,edgecolor='none',alpha=0.3)


# Inset plot
inset_ax = inset_axes(ax, width="56%", height="56%", bbox_to_anchor=(0.32, -0.07, 1.0, 1.0),
                      bbox_transform=ax.transAxes, loc='upper left')

inset_ax.set_ylim(0.5,0.9)
inset_ax.set_xlim(-1.0,-0.7)

inset_ax.plot(E1_E0_vec_qubit[0],E1_E0_vec_qubit[1],c=c1,marker='',label='EAPM $d=2$')

inset_ax.plot(E1_E0_vec_qutrit[0],E1_E0_vec_qutrit[1],c=c0,marker='',markersize=5,label='EAPM $d=3$')

inset_ax.plot(vecc,E1_PM_vec_up,c=c2)

inset_ax.fill_between(vecc, vecc, E1_PM_vec_up, color=c2,edgecolor='none',alpha=0.3)


# Draw lines from inset to zoom region on main plot
mark_inset(ax, inset_ax, loc1=3, loc2=1, fc="none", ec="0.5")


fig.legend(bbox_to_anchor=(0.6, 0.4),fontsize=10,frameon=False,ncols=1,loc='upper right')

plt.tight_layout()

# plt.savefig('E0_E1_EAPM.pdf')


















