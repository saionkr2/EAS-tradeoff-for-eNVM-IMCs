#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:45:35 2023

@author: saionroy
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy

Resnet_weights = np.load('resnet_cifar10_w.npy')
Resnet_bias = np.load('resnet_20_cifar10_biases.npy')
cifar10_labels = np.load('label_cifar10.npy')
Resnet_ideal_inputs = np.load('resnet_cifar10_x_unscaled.npy')
Resnet_weights[Resnet_weights==0] = -1

FIG_SIZE = (6,6)
GRID_ALPHA = 0.5
COLORS = {'r': 'r', 'g': 'lawngreen', 'b':'b', 'c': 'cyan', 'k': 'k', 'y':'y', 'm':'m', 'a':'salmon','l':'darkviolet','p':'teal'}
OUTPUT_DIR = 'figures'

colors = 'bgrckymalp'
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

#%%
#Evaluating Network Accuracy
NN_NOI = 10000
absolute_outputs = np.zeros([NN_NOI,10])
#absolute_outputs_Bw = np.zeros([NOI,10,4])
computed_labels = np.zeros([NN_NOI])
start_ind = 0
for i in range(start_ind,NN_NOI+start_ind):
    for j in range(10):
        for k in range(7):
            for l in [0,1]:
                if(k==0):
                    absolute_outputs[i-start_ind,j] = absolute_outputs[i-start_ind,j] - np.dot(Resnet_ideal_inputs[i,:,6-k],Resnet_weights[j,:,3-l])*2**(-k-1)*2**(-l-1) 
                else:
                    absolute_outputs[i-start_ind,j] = absolute_outputs[i-start_ind,j] + np.dot(Resnet_ideal_inputs[i,:,6-k],Resnet_weights[j,:,3-l])*2**(-k-1)*2**(-l-1) 
                               
    computed_labels[i-start_ind] = int(np.argmax(absolute_outputs[i-start_ind,:]*2+Resnet_bias))   
accuracy_fp = (np.sum(computed_labels == cifar10_labels[start_ind:])/NN_NOI)*100 
print(accuracy_fp) 

NOAE = 10
NOI = 64
N = 64
Bx = 7
Bw = 4
times = 100

#%%
B_ADC = 6   

predicted_weights_MSB = np.zeros([B_ADC,NOAE,NOI,times])
l2_error_MSB = np.zeros([B_ADC,NOAE,times])

predicted_weights_MSBm1 = np.zeros([B_ADC,NOAE,64,times])
l2_error_MSBm1 = np.zeros([B_ADC,NOAE,times])

reconstructed_unscaledflat_outputs = np.zeros([B_ADC,NN_NOI,times,NOAE])
reconstructed_unscaledflat_labels = np.zeros([B_ADC,NN_NOI,times])
reconstructed_w_unscaledflat = np.zeros([B_ADC,NOAE,N,times,2])
accuracy_reconstructed_unscaledflat = np.zeros([B_ADC,times])
mismatch_prob = np.zeros([B_ADC,times])

adc_ind = np.arange(66,66+40)
unscaled_measured_outputs = np.load('June2_2024_BasisAttack_16p6MHz_N64_ADC65-105.npz')
unscaled_measured_outputs = unscaled_measured_outputs[unscaled_measured_outputs.files[0]][adc_ind,:,:]

unscaled_measured_outputs = unscaled_measured_outputs*2-N  

starting_pt = 64
for ADC_prec in [1,2,3,4,5,6]:
#for ADC_prec in [1,6]:
    unscaled_MSB_outputs = np.zeros([NOAE,times])
    unscaled_MSB_outputs_6b = copy.deepcopy(unscaled_measured_outputs[0::4,:,:])

    if(ADC_prec!=6):
        unscaled_MSB_outputs = (unscaled_MSB_outputs_6b// (2 ** (6 - ADC_prec)) )* (2 ** (6 - ADC_prec))
    else:
        unscaled_MSB_outputs = unscaled_MSB_outputs_6b
    

    unscaled_measured_outputs_MSBrow = np.zeros([NOAE,NOI,times])
    for i in range(NOI):
        unscaled_measured_outputs_MSBrow[:,i,:] = unscaled_MSB_outputs[:,-1,:] - unscaled_MSB_outputs[:,i,:] 
        #unscaled_measured_outputs_MSBrow[:,i,:] = unscaled_MSB_outputs[:,i,:] - unscaled_MSB_outputs[:,-1,:]
 
    for i in range(NOAE):
        for j in range(times):
            predicted_weights_MSB[ADC_prec-1,i,:,j] = np.mean(unscaled_measured_outputs_MSBrow[i,:,:(j+1)],axis=1)
            predicted_weights_MSB[ADC_prec-1,i,predicted_weights_MSB[ADC_prec-1,i,:,j]<=0,j] = -1
            predicted_weights_MSB[ADC_prec-1,i,predicted_weights_MSB[ADC_prec-1,i,:,j]>0,j] = 1
            l2_error_MSB[ADC_prec-1,i,j] = np.linalg.norm(Resnet_weights[i,:,3]-predicted_weights_MSB[ADC_prec-1,i,:,j],ord=2)
            
    unscaled_MSBm1_outputs = np.zeros([NOAE,times])
    unscaled_MSBm1_outputs_6b = copy.deepcopy(unscaled_measured_outputs[2::4,:,:])

    if(ADC_prec!=6):
        unscaled_MSBm1_outputs = (unscaled_MSBm1_outputs_6b// (2 ** (6 - ADC_prec)) )* (2 ** (6 - ADC_prec))
    else:
        unscaled_MSBm1_outputs = unscaled_MSBm1_outputs_6b
    
    
    unscaled_measured_outputs_MSBm1row = np.zeros([NOAE,NOI,times])
    for i in range(NOI):
        unscaled_measured_outputs_MSBm1row[:,i,:] = unscaled_MSBm1_outputs[:,-1,:] - unscaled_MSBm1_outputs[:,i,:] 
        #unscaled_measured_outputs_MSBm1row[:,i,:] = unscaled_MSBm1_outputs[:,i,:] - unscaled_MSBm1_outputs[:,-1,:]

    for i in range(NOAE):
        for j in range(times):
            predicted_weights_MSBm1[ADC_prec-1,i,:,j] = np.mean(unscaled_measured_outputs_MSBm1row[i,:,:(j+1)],axis=1)
            predicted_weights_MSBm1[ADC_prec-1,i,predicted_weights_MSBm1[ADC_prec-1,i,:,j]<=0,j] = -1
            predicted_weights_MSBm1[ADC_prec-1,i,predicted_weights_MSBm1[ADC_prec-1,i,:,j]>0,j] = 1
            l2_error_MSBm1[ADC_prec-1,i,j] = np.linalg.norm(Resnet_weights[i,:,2]-predicted_weights_MSBm1[ADC_prec-1,i,:,j],ord=2)

    #Network-level attack metrics
    reconstructed_w_unscaledflat[ADC_prec-1,:,:,:,1] = predicted_weights_MSB[ADC_prec-1,:,:,:]
    reconstructed_w_unscaledflat[ADC_prec-1,:,:,:,0] = predicted_weights_MSBm1[ADC_prec-1,:,:,:]

    for i in range(NN_NOI):
        for j in range(10):
            for k in range(7):
                for l in [0,1]:
                    if(k==0):
                        reconstructed_unscaledflat_outputs[ADC_prec-1,i,:,j] = reconstructed_unscaledflat_outputs[ADC_prec-1,i,:,j] - np.matmul(Resnet_ideal_inputs[i,:,6-k],reconstructed_w_unscaledflat[ADC_prec-1,j,:,:,1-l])*2**(-k-1)*2**(-l-1) 
                    else:   
                        reconstructed_unscaledflat_outputs[ADC_prec-1,i,:,j] = reconstructed_unscaledflat_outputs[ADC_prec-1,i,:,j] + np.matmul(Resnet_ideal_inputs[i,:,6-k],reconstructed_w_unscaledflat[ADC_prec-1,j,:,:,1-l])*2**(-k-1)*2**(-l-1) 
                           
        reconstructed_unscaledflat_labels[ADC_prec-1,i,:] = (np.argmax(reconstructed_unscaledflat_outputs[ADC_prec-1,i,:,:]*2+Resnet_bias,axis=1)).astype(int)   

    for i in range(times):
        accuracy_reconstructed_unscaledflat[ADC_prec-1,i] = (np.sum(reconstructed_unscaledflat_labels[ADC_prec-1,:,i] == cifar10_labels)/NN_NOI)*100 
        mismatch_prob[ADC_prec-1,i] = (np.sum(computed_labels != reconstructed_unscaledflat_labels[ADC_prec-1,:,i])/NN_NOI)*100
        

weight_norm_MSB = np.zeros(NOAE)
weight_norm_MSBm1 = np.zeros(NOAE)
for i in range(NOAE):
    weight_norm_MSB[i] = np.linalg.norm(Resnet_weights[i,:,3], ord=2)
    weight_norm_MSBm1[i] = np.linalg.norm(Resnet_weights[i,:,2], ord=2)
    
np.save('l2_error_MSB_BasisAtack_16p6.npy',l2_error_MSB)
np.save('l2_error_MSBm1_BasisAtack_16p6.npy',l2_error_MSBm1)
np.save('inference_accuracy_BasisAtack_16p6.npy',accuracy_reconstructed_unscaledflat)
np.save('mismatch_prob_BasisAtack_16p6.npy',mismatch_prob)

#%%       
loc_idx = 0
ftsize = 15
fig, ax = plt.subplots(figsize = FIG_SIZE)
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
ax.set_xlabel(r'# of Measurements $M$', fontsize = 25,fontdict=font)
ax.set_ylabel(r'Bit Error Rate (%)', fontsize = 25,fontdict=font)
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=15) 
#ax.set_xscale('log',base=10)
starting_pt = 1
ADCid = 1
for ADC_prec in range(1,7):
    ax.plot(np.arange(starting_pt,times),100*l2_error_MSB[ADC_prec-1,ADCid,starting_pt:times]**2/(weight_norm_MSB[ADCid]**2*4), markerfacecolor='none', ms=18)

#plt.legend(frameon=True,framealpha=1)
#plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.2),prop={'size': 22}                 
#            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)

# plt.savefig('BER_MSB_BasisAttack_8p3.pdf',bbox_inches='tight')

#%%       
loc_idx = 0
ftsize = 15
fig, ax = plt.subplots(figsize = FIG_SIZE)
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
ax.set_xlabel(r'# of Measurements $M$', fontsize = 25,fontdict=font)
ax.set_ylabel(r'Bit Error Rate (%)', fontsize = 25,fontdict=font)
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=15) 
#ax.set_xscale('log',base=10)
starting_pt = 1
ADCid = 9
for ADC_prec in range(1,7):
    ax.plot(np.arange(starting_pt,times),100*l2_error_MSBm1[ADC_prec-1,ADCid,starting_pt:times]**2/(weight_norm_MSBm1[ADCid]**2*4), markerfacecolor='none', ms=18)
    
#plt.legend(frameon=True,framealpha=1)
#plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.2),prop={'size': 22}                 
#            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)

# plt.savefig('BER_MSBm1_BasisAttack_8p3.pdf',bbox_inches='tight')

#%%       
loc_idx = 0
ftsize = 15
fig, ax = plt.subplots(figsize = FIG_SIZE)
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
ax.set_xlabel(r'# of Measurements $M$', fontsize = 25,fontdict=font)
ax.set_ylabel(r'CIFAR-10 accuracy (%)', fontsize = 25,fontdict=font)
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=15) 
#ax.set_xscale('log',base=10)
starting_pt = 1
for ADC_prec in range(1,7):
    ax.plot(np.arange(starting_pt,times),accuracy_reconstructed_unscaledflat[ADC_prec-1,starting_pt:times], markerfacecolor='none', ms=18)

# plt.savefig('CIFAR-10_Acc_BasisAttack_8p3.pdf',bbox_inches='tight')

#%%       
loc_idx = 0
ftsize = 15
fig, ax = plt.subplots(figsize = FIG_SIZE)
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
ax.set_xlabel(r'# of Measurements $M$', fontsize = 25,fontdict=font)
ax.set_ylabel(r'mismatch probability $p_m$ (%)', fontsize = 25,fontdict=font)
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=15) 
#ax.set_xscale('log',base=10)
starting_pt = 1
ADCid = 0
for ADC_prec in range(1,7):
    ax.plot(np.arange(starting_pt,times),mismatch_prob[ADC_prec-1,starting_pt:times], markerfacecolor='none', ms=18)

# plt.savefig('CIFAR-10_Acc_BasisAttack_8p3.pdf',bbox_inches='tight')