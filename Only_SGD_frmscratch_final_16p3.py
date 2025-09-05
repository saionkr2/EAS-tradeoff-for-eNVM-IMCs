# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:18:42 2024

@author: saionkr2
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import copy
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    #return array[idx]

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

import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

Resnet_weights = np.load('resnet_cifar10_w.npy')
Resnet_weights[Resnet_weights==0] = -1

N=64
NOI = 10000
NOAE = 10
times = 10

ADC_npz_codeset_v1 = np.load('May28_2024_OLSAttack_100k_10times_16p6MHz_N64_ADC65-105.npz')
adc_read_codeset_v1 = ADC_npz_codeset_v1[ADC_npz_codeset_v1.files[0]]*2-N

adc_read_codeset = np.zeros([128,NOI,times])
adc_read_codeset[:,:10000,:] = copy.deepcopy(adc_read_codeset_v1)

Binary_WL_inputs_v1 = np.load('Input_col.npy')

Binary_WL_inputs = np.zeros([128,NOI])
Binary_WL_inputs[:,:10000] = copy.deepcopy(Binary_WL_inputs_v1)

Binary_unscaled_inputs = Binary_WL_inputs[0::2,:]

Binary_unscaled_inputs[Binary_unscaled_inputs==0] = -1

Binary_unscaled_inputs = Binary_unscaled_inputs.T

FIG_SIZE = (6,6)
GRID_ALPHA = 0.5
COLORS = {'r': 'r', 'g': 'lawngreen', 'b':'b', 'c': 'cyan', 'k': 'k', 'y':'y', 'm':'m', 'a':'salmon','l':'darkviolet','p':'teal'}
OUTPUT_DIR = 'figures'

colors = 'bgrckymalp'
marker = 'odos'
marker_size = 100 

ier = 64

adc_ind = np.arange(66,66+40)
ADC_selected = copy.deepcopy(adc_read_codeset[adc_ind,:,:])
ADC_selected_MSB = ADC_selected[0::4,:,:]
ADC_selected_MSBm1 = ADC_selected[2::4,:,:]
ADC_temp = copy.deepcopy(adc_read_codeset[adc_ind,:,:])

#%%

Resnet_weights = np.load('resnet_cifar10_w.npy')
Resnet_bias = np.load('resnet_20_cifar10_biases.npy')
cifar10_labels = np.load('label_cifar10.npy')
Resnet_ideal_inputs = np.load('resnet_cifar10_x_unscaled.npy')
Resnet_weights[Resnet_weights==0] = -1

#Evaluating Network Accuracy
NN_NOI = 10000
absolute_outputs = np.zeros([NN_NOI,10])
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

TE = 10 #Total number of epochs
#%%
def isNaN(num):
    return num != num

def mean_square_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def ber(w_true, w_pred):
    l2_error = np.linalg.norm(w_true - w_pred,2)
    weight_norm = np.linalg.norm(w_true,2)
    return 100*l2_error**2/(weight_norm**2*4)

def gradient(w, x, y_true):
    # Compute gradient of the loss function with respect to w
    #x_ones = np.ones(64)
    y_pred =  np.dot(x, w) #+ np.dot(x_ones,np.multiply(a,w**2)) + np.dot(x,np.multiply(b,w**3))
    error = y_true - y_pred
    #print(y_pred,a,np.dot(x, w))
    grad_w = -2 * np.dot(x.T, error) #/ len(y_true)
    return grad_w

def stochastic_gradient_descent(x, y, learning_rate=0.0001, epochs=TE):
    # Initialize weights randomly
    # w = np.random.randint(2, size=(x.shape[1])).astype('float64')
    w = np.random.randn(x.shape[1])
    #a = np.random.randn(x.shape[1])
    #b = np.random.randn(x.shape[1])
    #print(x.shape[1])
    #x_ones = np.ones(64)
    w_est = np.zeros([64,x.shape[0]*epochs])
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(y))
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        # Update weights for each data point
        for i in range(len(y)):
        #for i in range(5):
            grad_w = gradient(w, x_shuffled[i], y_shuffled[i])
            w -= learning_rate * grad_w
            #a -= learning_rate * grad_a
            #b -= learning_rate * grad_b
            w_est[:,epoch*len(y)+i] = w    
        # Print loss every epoch
        y_pred = np.dot(x, w) #+ np.dot(x_ones,np.multiply(a,w**2)) + np.dot(x,np.multiply(b,w**3))
        loss = mean_square_loss(y, y_pred)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
    w_est = w_est[:,TE-1::TE]   
    return w_est
#%%
B_ADC = 6
NOI_attack = 2000
times_attack = 5
times_calib = 5
predicted_weights_MSB = np.zeros([B_ADC,NOAE,64,NOI_attack,times_attack,times_calib])
predicted_weights_MSB_avg = np.zeros([B_ADC,NOAE,64,NOI_attack])
l2_error_MSB_avg = np.zeros([B_ADC,NOAE,NOI_attack])

predicted_weights_MSBm1 = np.zeros([B_ADC,NOAE,64,NOI_attack,times_attack,times_calib])
predicted_weights_MSBm1_avg = np.zeros([B_ADC,NOAE,64,NOI_attack])
l2_error_MSBm1_avg = np.zeros([B_ADC,NOAE,NOI_attack])

reconstructed_unscaledflat_outputs = np.zeros([B_ADC,NN_NOI,NOI_attack,10])
reconstructed_unscaledflat_labels = np.zeros([B_ADC,NN_NOI,NOI_attack])
reconstructed_w_unscaledflat = np.zeros([B_ADC,10,N,NOI_attack,2])
accuracy_reconstructed_unscaledflat = np.zeros([B_ADC,NOI_attack])
mismatch_prob = np.zeros([B_ADC,NOI_attack])

starting_pt = 1

for ADC_prec in [1,2,3,4,5,6]:
#for ADC_prec in [1,6]:
    for ta in range(times_attack):
        unscaled_MSB_outputs = np.zeros([NOAE,NOI_attack,times_calib])
        unscaled_measured_outputs_6b = copy.deepcopy(ADC_selected_MSB[:,ta*NOI_attack:(ta+1)*NOI_attack,:times_calib])

        if(ADC_prec!=6):
            unscaled_MSB_outputs = (unscaled_measured_outputs_6b// (2 ** (6 - ADC_prec)) )* (2 ** (6 - ADC_prec))
        else:
            unscaled_MSB_outputs = unscaled_measured_outputs_6b
        
               
        MSB_training_set = unscaled_MSB_outputs
        
        for i in range(NOAE):
            for k in range(times_calib):
                print(i,k)
                predicted_weights_MSB[ADC_prec-1,i,:,:,ta,k] = stochastic_gradient_descent(Binary_unscaled_inputs[ta*NOI_attack:(ta+1)*NOI_attack,:], MSB_training_set[i,:,k])
          
        unscaled_MSBm1_outputs = np.zeros([NOAE,NOI_attack,times_calib])
        unscaled_measured_outputs_6b = copy.deepcopy(ADC_selected_MSBm1[:,ta*NOI_attack:(ta+1)*NOI_attack,:times_calib])

        if(ADC_prec!=6):
            unscaled_MSBm1_outputs = (unscaled_measured_outputs_6b// (2 ** (6 - ADC_prec)) )* (2 ** (6 - ADC_prec))
        else:
            unscaled_MSBm1_outputs = unscaled_measured_outputs_6b
        
       
        MSBm1_training_set = unscaled_MSBm1_outputs
        
        for i in range(NOAE):
            for k in range(times_calib):
                print(i,k)
                predicted_weights_MSBm1[ADC_prec-1,i,:,:,ta,k] = stochastic_gradient_descent(Binary_unscaled_inputs[ta*NOI_attack:(ta+1)*NOI_attack,:], MSBm1_training_set[i,:,k])
    
    #Time average followed by ensemble average
    predicted_weights_MSB_avg[ADC_prec-1,:,:,:] = np.mean(np.mean(predicted_weights_MSB[ADC_prec-1,:,:,:,:,:], axis=4), axis=3)
    #predicted_weights_MSB_avg[ADC_prec-1,:,:,:] = np.mean(predicted_weights_MSB[ADC_prec-1,:,:,:,0,:], axis=3)
    predicted_weights_MSB_avg[ADC_prec-1,predicted_weights_MSB_avg[ADC_prec-1,:,:,:]<=0] = -1
    predicted_weights_MSB_avg[ADC_prec-1,predicted_weights_MSB_avg[ADC_prec-1,:,:,:]>0] = 1
        
    for i in range(NOAE):
        for j in range(NOI_attack):
            l2_error_MSB_avg[ADC_prec-1,i,j] = np.linalg.norm(Resnet_weights[i,:,3]-predicted_weights_MSB_avg[ADC_prec-1,i,:,j],ord=2)
    
    #Time average followed by ensemble average
    predicted_weights_MSBm1_avg[ADC_prec-1,:,:,:] = np.mean(np.mean(predicted_weights_MSBm1[ADC_prec-1,:,:,:,:,:], axis=4), axis=3)
    #predicted_weights_MSBm1_avg[ADC_prec-1,:,:,:] = np.mean(predicted_weights_MSBm1[ADC_prec-1,:,:,:,0,:], axis=3)
    predicted_weights_MSBm1_avg[ADC_prec-1,predicted_weights_MSBm1_avg[ADC_prec-1,:,:,:]<=0] = -1
    predicted_weights_MSBm1_avg[ADC_prec-1,predicted_weights_MSBm1_avg[ADC_prec-1,:,:,:]>0] = 1
        
    for i in range(NOAE):
        for j in range(NOI_attack):
            l2_error_MSBm1_avg[ADC_prec-1,i,j] = np.linalg.norm(Resnet_weights[i,:,2]-predicted_weights_MSBm1_avg[ADC_prec-1,i,:,j],ord=2)

    print(ADC_prec)
    
    #Network-level attack metrics
    reconstructed_w_unscaledflat[ADC_prec-1,:,:,:,1] = predicted_weights_MSB_avg[ADC_prec-1,:,:,:]
    reconstructed_w_unscaledflat[ADC_prec-1,:,:,:,0] = predicted_weights_MSBm1_avg[ADC_prec-1,:,:,:]

    for i in range(NN_NOI):
        for j in range(10):
            for k in range(7):
                for l in [0,1]:
                    if(k==0):
                        reconstructed_unscaledflat_outputs[ADC_prec-1,i,:,j] = reconstructed_unscaledflat_outputs[ADC_prec-1,i,:,j] - np.matmul(Resnet_ideal_inputs[i,:,6-k],reconstructed_w_unscaledflat[ADC_prec-1,j,:,:,1-l])*2**(-k-1)*2**(-l-1) 
                    else:   
                        reconstructed_unscaledflat_outputs[ADC_prec-1,i,:,j] = reconstructed_unscaledflat_outputs[ADC_prec-1,i,:,j] + np.matmul(Resnet_ideal_inputs[i,:,6-k],reconstructed_w_unscaledflat[ADC_prec-1,j,:,:,1-l])*2**(-k-1)*2**(-l-1) 
                           
        reconstructed_unscaledflat_labels[ADC_prec-1,i,:] = (np.argmax(reconstructed_unscaledflat_outputs[ADC_prec-1,i,:,:]*2+Resnet_bias,axis=1)).astype(int)   

    for i in range(NOI_attack):
        accuracy_reconstructed_unscaledflat[ADC_prec-1,i] = (np.sum(reconstructed_unscaledflat_labels[ADC_prec-1,:,i] == cifar10_labels)/NN_NOI)*100 
        mismatch_prob[ADC_prec-1,i] = (np.sum(computed_labels != reconstructed_unscaledflat_labels[ADC_prec-1,:,i])/NN_NOI)*100

np.save('l2_error_MSB_SDG_lin_16p6.npy',l2_error_MSB_avg)
np.save('l2_error_MSBm1_SDG_lin_16p6.npy',l2_error_MSBm1_avg)
np.save('inference_accuracy_SDG_lin_16p6.npy',accuracy_reconstructed_unscaledflat)
np.save('mismatch_prob_SDG_lin_16p6.npy',mismatch_prob)

weight_norm_MSB = np.zeros(NOAE)
weight_norm_MSBm1 = np.zeros(NOAE)
for i in range(NOAE):
    weight_norm_MSB[i] = np.linalg.norm(Resnet_weights[i,:,3], ord=2)
    weight_norm_MSBm1[i] = np.linalg.norm(Resnet_weights[i,:,2], ord=2)
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
starting_pt = 64
ADCid = 0
for ADC_prec in range(1,7):
    ax.plot(np.arange(starting_pt,NOI_attack),100*l2_error_MSB_avg[ADC_prec-1,ADCid,starting_pt:NOI_attack]**2/(weight_norm_MSB[ADCid]**2*4), markerfacecolor='none', ms=18)
    
#plt.legend(frameon=True,framealpha=1)
#plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.2),prop={'size': 22}                 
#            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)

# plt.savefig('BER_MSB_SDG_lin_16p6.pdf',bbox_inches='tight')

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
starting_pt = 64
ADCid = 0
for ADC_prec in range(1,7):
    ax.plot(np.arange(starting_pt,NOI_attack),100*l2_error_MSBm1_avg[ADC_prec-1,ADCid,starting_pt:NOI_attack]**2/(weight_norm_MSBm1[ADCid]**2*4), markerfacecolor='none', ms=18)
    
#plt.legend(frameon=True,framealpha=1)
#plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.2),prop={'size': 22}                 
#            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)

# plt.savefig('BER_MSBm1_SDG_lin_16p6.pdf',bbox_inches='tight')

#%%       
accuracy_reconstructed_unscaledflat = np.load('inference_accuracy_SDG_lin_16p6.npy')
NOI_attack = 2000   
loc_idx = 0
ftsize = 15
fig, ax = plt.subplots(figsize = FIG_SIZE)
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
ax.set_xlabel(r'# of Measurements $M$', fontsize = 25,fontdict=font)
ax.set_ylabel(r'CIFAR-10 accuracy (%)', fontsize = 25,fontdict=font)
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=15) 
#ax.set_xscale('log',base=10)
starting_pt = 64
for ADC_prec in range(1,7):
    ax.plot(np.arange(starting_pt,NOI_attack),accuracy_reconstructed_unscaledflat[ADC_prec-1,starting_pt:NOI_attack], markerfacecolor='none', ms=18)

# plt.savefig('CIFAR-10_Acc_SDG_lin_16p6.pdf',bbox_inches='tight')

#%%     
mismatch_prob = np.load('mismatch_prob_SDG_lin_16p6.npy')   
loc_idx = 0
ftsize = 15
fig, ax = plt.subplots(figsize = FIG_SIZE)
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
ax.set_xlabel(r'# of Measurements $M$', fontsize = 25,fontdict=font)
ax.set_ylabel(r'mismatch probability $p_m$ (%)', fontsize = 25,fontdict=font)
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=15) 
#ax.set_xscale('log',base=10)
starting_pt = 64
ADCid = 0
for ADC_prec in range(1,7):
    ax.plot(np.arange(starting_pt,NOI_attack),mismatch_prob[ADC_prec-1,starting_pt:NOI_attack], markerfacecolor='none', ms=18)

# plt.savefig('CIFAR-10_Acc_SDG_lin_16p6.pdf',bbox_inches='tight')