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


SNDR_MSBall_8p3 = np.load('SNDR_all_8p3_MSB.npy')
SNDR_MSBall_8p3LB = np.load('SNDR_all_8p3LB_MSB.npy')
SNDR_MSBall_16p6 = np.load('SNDR_all_16p6_MSB.npy')
SNDR_MSBall_16p6LB = np.load('SNDR_all_16p6LB_MSB.npy')

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

ADC_best = 3
ADC_worst = 7
ADC_typ = 4

NOI = 2000
# ADC_best = 1
# ADC_worst = 6
# ADC_typ = 2

#%%
ftsize = 15
fig, ax = plt.subplots(figsize =(6,6))
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted', linewidth=0.3)
#ax.set_ylim([0,14.5])
ax.set_xlim([50,800])

#ax.set_title('SNR with Random and per-code inputs', fontsize = 25, fontweight = 'bold', family ='calibri')
ax.set_xlabel(r'$E_{\mathrm{op1}}$ (fJ)', fontsize = 18)
ax.set_ylabel(r'$\mathrm{SNDR}_{\mathrm{d}}$ (dB)', fontsize = 18)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
msize=10
#plt.xticks(np.arange(NOAE)+1)
ax.plot(energy8p3, SNDR_all_8p3[:,5], "s-", color=colors[0], markersize=msize, label='HB-LF')
ax.plot(energy8p3LB, SNDR_all_8p3LB[:,5], "s-", color=colors[1], markersize=msize, label='LB-LF')
ax.plot(energy16p6[:], SNDR_all_16p6[:,5], "s-", color=colors[2], markersize=msize, label='HB-HF')
ax.plot(energy16p6LB[:], SNDR_all_16p6LB[:,5], "s-", color=colors[3], markersize=msize, label='LB-HF')

ax.legend(loc='best', ncol=1,prop={'size': 20, 'family':'Arial'}                        
            ,edgecolor='black',columnspacing=0.5,handlelength=1.0,handletextpad=0.5) 
#ax.plot(DP_in, ADC_average[loc_idx,:], markerfacecolor='none', ms=18, markeredgecolor='red', markeredgewidth = 1.5 )
fig.savefig('SNDR_col_worst_vs_Energy_Extended.pdf', bbox_inches = "tight",dpi=500)

#%%
ftsize = 15
fig, ax = plt.subplots(figsize =(6,6))
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted', linewidth=0.3)
#ax.set_ylim([0,14.5])
ax.set_xlim([50,800])

#ax.set_title('SNR with Random and per-code inputs', fontsize = 25, fontweight = 'bold', family ='calibri')
ax.set_xlabel(r'$E_{\mathrm{op1}}$ (fJ)', fontsize = 18)
ax.set_ylabel(r'$\mathrm{SNDR}_{\mathrm{d}}$ (dB)', fontsize = 18)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
msize=10
#plt.xticks(np.arange(NOAE)+1)
ax.plot(energy8p3, SNDR_all_8p3[:,7], "s-", color=colors[0], markersize=msize, label='HB-LF')
ax.plot(energy8p3LB, SNDR_all_8p3LB[:,7], "s-", color=colors[1], markersize=msize, label='LB-LF')
ax.plot(energy16p6[:], SNDR_all_16p6[:,7], "s-", color=colors[2], markersize=msize, label='HB-HF')
ax.plot(energy16p6LB[:], SNDR_all_16p6LB[:,7], "s-", color=colors[3], markersize=msize, label='LB-HF')

ax.legend(loc='best', ncol=1,prop={'size': 20, 'family':'Arial'}                        
            ,edgecolor='black',columnspacing=0.5,handlelength=1.0,handletextpad=0.5) 
#ax.plot(DP_in, ADC_average[loc_idx,:], markerfacecolor='none', ms=18, markeredgecolor='red', markeredgewidth = 1.5 )
fig.savefig('SNDR_col_best_vs_Energy_Extended.pdf', bbox_inches = "tight",dpi=500)

#%%
ftsize = 15
fig, ax = plt.subplots(figsize =(6,6))
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted', linewidth=0.3)
#ax.set_ylim([0,14.5])
ax.set_xlim([50,800])

#ax.set_title('SNR with Random and per-code inputs', fontsize = 25, fontweight = 'bold', family ='calibri')
ax.set_xlabel(r'$E_{\mathrm{op1}}$ (fJ)', fontsize = 18)
ax.set_ylabel(r'average $\mathrm{SNDR}_{\mathrm{d}}$ (dB)', fontsize = 18)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
msize=10
#plt.xticks(np.arange(NOAE)+1)
ax.plot(energy8p3, SNDR_IMC_8p3, "s-", color=colors[0], markersize=msize, label='HB-LF')
ax.plot(energy8p3LB, SNDR_IMC_8p3LB, "s-", color=colors[1], markersize=msize, label='LB-LF')
ax.plot(energy16p6[:], SNDR_IMC_16p6, "s-", color=colors[2], markersize=msize, label='HB-HF')
ax.plot(energy16p6LB[:], SNDR_IMC_16p6LB, "s-", color=colors[3], markersize=msize, label='LB-HF')

ax.legend(loc='best', ncol=1,prop={'size': 20, 'family':'Arial'}                        
            ,edgecolor='black',columnspacing=0.5,handlelength=1.0,handletextpad=0.5) 
#ax.plot(DP_in, ADC_average[loc_idx,:], markerfacecolor='none', ms=18, markeredgecolor='red', markeredgewidth = 1.5 )
fig.savefig('SNDR_IMC_vs_Energy_Extended.pdf', bbox_inches = "tight",dpi=500)

#%%
ftsize = 15
fig, ax = plt.subplots(figsize =(6,6))
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted', linewidth=0.3)
#ax.set_ylim([0,14.5])
ax.set_xlim([50,800])

#ax.set_title('SNR with Random and per-code inputs', fontsize = 25, fontweight = 'bold', family ='calibri')
ax.set_xlabel(r'$E_{\mathrm{op1}}$ (fJ)', fontsize = 18)
ax.set_ylabel(r'$\mathrm{SNDR}_{\mathrm{d}}$ (dB)', fontsize = 18)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
msize=10
#plt.xticks(np.arange(NOAE)+1)
for i in range(9):
    ax.plot(energy8p3, SNDR_all_8p3[:,i], "s-", color=colors[0], markersize=msize)
    ax.plot(energy16p6[:], SNDR_all_16p6[:,i], "s-", color=colors[2], markersize=msize)
    ax.plot(energy8p3LB, SNDR_all_8p3LB[:,i], "s-", color=colors[1], markersize=msize)
    ax.plot(energy16p6LB[:], SNDR_all_16p6LB[:,i], "s-", color=colors[3], markersize=msize)
    
ax.plot(energy8p3, SNDR_all_8p3[:,9], "s-", color=colors[0], markersize=msize, label='HB-LF')
ax.plot(energy16p6[:], SNDR_all_16p6[:,9], "s-", color=colors[2], markersize=msize, label='HB-HF')
ax.plot(energy8p3LB, SNDR_all_8p3LB[:,9], "s-", color=colors[1], markersize=msize, label='LB-LF')
ax.plot(energy16p6LB[:], SNDR_all_16p6LB[:,9], "s-", color=colors[3], markersize=msize, label='LB-HF')

ax.legend(loc='best', ncol=1,prop={'size': 20, 'family':'Arial'}                        
            ,edgecolor='black',columnspacing=0.5,handlelength=1.0,handletextpad=0.5) 
#ax.plot(DP_in, ADC_average[loc_idx,:], markerfacecolor='none', ms=18, markeredgecolor='red', markeredgewidth = 1.5 )
fig.savefig('SNDR_col_all_vs_Energy_Extended.pdf', bbox_inches = "tight",dpi=500)

#%%
ftsize = 15
fig, ax = plt.subplots(figsize =(6,6))
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted', linewidth=0.3)
#ax.set_ylim([0,14.5])
ax.set_xlim([50,800])

#ax.set_title('SNR with Random and per-code inputs', fontsize = 25, fontweight = 'bold', family ='calibri')
ax.set_xlabel(r'$E_{\mathrm{op1}}$ (fJ)', fontsize = 18)
ax.set_ylabel(r'$\mathrm{SNDR}_{\mathrm{d}}$ (dB)', fontsize = 18)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
msize=10
#plt.xticks(np.arange(NOAE)+1)
for i in range(9):
    ax.plot(energy8p3, SNDR_MSBall_8p3[:,i], "s-", color=colors[0], markersize=msize)
    ax.plot(energy16p6[:], SNDR_MSBall_16p6[:,i], "s-", color=colors[2], markersize=msize)
    ax.plot(energy8p3LB, SNDR_MSBall_8p3LB[:,i], "s-", color=colors[1], markersize=msize)
    ax.plot(energy16p6LB[:], SNDR_MSBall_16p6LB[:,i], "s-", color=colors[3], markersize=msize)
    
ax.plot(energy8p3, SNDR_MSBall_8p3[:,9], "s-", color=colors[0], markersize=msize, label='HB-LF')
ax.plot(energy16p6[:], SNDR_MSBall_16p6[:,9], "s-", color=colors[2], markersize=msize, label='HB-HF')
ax.plot(energy8p3LB, SNDR_MSBall_8p3LB[:,9], "s-", color=colors[1], markersize=msize, label='LB-LF')
ax.plot(energy16p6LB[:], SNDR_MSBall_16p6LB[:,9], "s-", color=colors[3], markersize=msize, label='LB-HF')

ax.legend(loc='best', ncol=1,prop={'size': 20, 'family':'Arial'}                        
            ,edgecolor='black',columnspacing=0.5,handlelength=1.0,handletextpad=0.5) 
#ax.plot(DP_in, ADC_average[loc_idx,:], markerfacecolor='none', ms=18, markeredgecolor='red', markeredgewidth = 1.5 )
#fig.savefig('SNDR_col_all_vs_Energy_Extended.pdf', bbox_inches = "tight",dpi=500)