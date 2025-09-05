#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:45:35 2023

@author: saionroy
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

FIG_SIZE = (6,6)
GRID_ALPHA = 0.5
COLORS = {'r': 'r', 'g': 'lawngreen', 'b':'b', 'c': 'cyan', 'k': 'k', 'y':'y', 'm':'m', 'a':'salmon','l':'darkviolet','p':'teal'}
OUTPUT_DIR = 'figures'

colors =  ['black','limegreen','deepskyblue','red']
marker = 'odos'
marker_size = 100 

font = {'family': 'arial',
        'color':  'black',
        'weight': 'regular',
        'size': 16,
        }

matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'

SNDR_all_8p3 = np.load('SNDR_all_8p3_MSBm1.npy')
SNDR_IMC_8p3 = np.load('SNDR_IMC_8p3_MSBm1.npy')
energy8p3 = np.load('Energy_8p3.npy')

SNDR_all_8p3LB = np.load('SNDR_all_8p3LB_MSBm1.npy')
SNDR_IMC_8p3LB = np.load('SNDR_IMC_8p3LB_MSBm1.npy')
energy8p3LB = np.load('Energy_8p3LB.npy')

SNDR_all_16p6 = np.load('SNDR_all_16p6_MSBm1.npy')
SNDR_IMC_16p6 = np.load('SNDR_IMC_16p6_MSBm1.npy')
energy16p6 = np.load('Energy_16p6.npy')

SNDR_all_16p6LB = np.load('SNDR_all_16p6LB_MSBm1.npy')
SNDR_IMC_16p6LB = np.load('SNDR_IMC_16p6LB_MSBm1.npy')
energy16p6LB = np.load('Energy_16p6LB.npy')

            
inference_accuacy_BasisAtack_8p3 = np.load('inference_accuracy_BasisAtack_8p3.npy')
inference_accuacy_BasisAtack_8p3_max = np.max(inference_accuacy_BasisAtack_8p3,axis=1)
inference_accuacy_BasisAtack_8p3_argmax = np.argmax(inference_accuacy_BasisAtack_8p3,axis=1)


inference_accuacy_BasisAtack_8p3LB = np.load('inference_accuracy_BasisAtack_8p3LB.npy')
inference_accuacy_BasisAtack_8p3LB_max = np.max(inference_accuacy_BasisAtack_8p3LB,axis=1)
inference_accuacy_BasisAtack_8p3LB_argmax = np.argmax(inference_accuacy_BasisAtack_8p3LB,axis=1)

inference_accuacy_BasisAtack_16p6 = np.load('inference_accuracy_BasisAtack_16p6.npy')
inference_accuacy_BasisAtack_16p6_max = np.max(inference_accuacy_BasisAtack_16p6,axis=1)
inference_accuacy_BasisAtack_16p6_argmax = np.argmax(inference_accuacy_BasisAtack_16p6,axis=1)


inference_accuacy_BasisAtack_16p6LB = np.load('inference_accuracy_BasisAtack_16p6LB.npy')
inference_accuacy_BasisAtack_16p6LB_max = np.max(inference_accuacy_BasisAtack_16p6LB,axis=1)
inference_accuacy_BasisAtack_16p6LB_argmax = np.argmax(inference_accuacy_BasisAtack_16p6LB,axis=1)


#%%
#Packages Used
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import matplotlib.ticker as mticker
   
def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"

#For storing the columnwise values into their respective X, Y and Z values

x1 = energy8p3
x2 = energy8p3LB
x3 = energy16p6
x4 = energy16p6LB

y1 = SNDR_IMC_8p3
y2 = SNDR_IMC_8p3LB
y3 = SNDR_IMC_16p6
y4 = SNDR_IMC_16p6LB

z1 = inference_accuacy_BasisAtack_8p3_max
z2 = inference_accuacy_BasisAtack_8p3LB_max
z3 = inference_accuacy_BasisAtack_16p6_max
z4 = inference_accuacy_BasisAtack_16p6LB_max

rcParams['axes.labelpad'] = 10
rcParams['axes.linewidth'] = 1.5
font = {'family': 'arial',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

c_list = []
for c in colors[0]:
    c_list.extend(['b'] * len(x1))
    
fig = plt.figure(figsize=(11,8))
ax = plt.axes(projection='3d')
#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                cmap='viridis', edgecolor='none')
ax.set_ylim(0,4)
ax.set_zlim(-5,95)
ax.plot3D(x1, y1, z1, 'o-', markerfacecolor=colors[0], ms=10, label=r'HB-LF',color=colors[0])
ax.plot3D(x3, y3, z3, 'o-', markerfacecolor=colors[2], ms=10, label=r'HB-HF',color=colors[2])
ax.plot3D(x2, y2, z2, 'o-', markerfacecolor=colors[1], ms=10, label=r'LB-LF',color=colors[1])
ax.plot3D(x4, y4, z4, 'o-', markerfacecolor=colors[3], ms=10, label=r'LB-HF',color=colors[3])
ax.scatter(x1, y1, zdir='z', c=colors[0],alpha=1,s=100)
ax.scatter(x2, y2, zdir='z', c=colors[1],alpha=1,s=100)
ax.scatter(x3, y3, zdir='z', c=colors[2],alpha=1,s=100)
ax.scatter(x4, y4, zdir='z', c=colors[3],alpha=1,s=100)

tick_size = 14
label_size = 16
ax.locator_params(axis="x", nbins=6)
ax.locator_params(axis="y", nbins=6)
ax.tick_params(size=tick_size, labelsize=label_size)
ax.text(0.18, 1.06, 1, r'Attack',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=20, fontname='Arial')    
plt.legend(loc='upper center', ncol=4,bbox_to_anchor=(0.55,0.97),prop={'size': 20, 'family':'Arial'}                        
            ,edgecolor='black',columnspacing=0.5,handlelength=1.0,handletextpad=0.5) 
#ax.set_xticklabels(labels,fontdict=font, minor=True)
#All: Elevation: 30, Azimuthal angle: 60
ax.view_init(15,-45)
#fig.colorbar(surf, shrink =0.6, aspect =18, pad = 0.05)

plt.savefig('BasisAtack_3D_plot.pdf', bbox_inches='tight',transparent=True)
plt.show()
