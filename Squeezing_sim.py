import math
from math import pi, sqrt, exp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import random
from optparse import OptionParser
import csv
import os
from scipy.stats import norm


def DFT(a):
    
    a_sum = 0
    for i in range(len(a)):
        a_sum += a[i] * np.exp(1j * 2*pi * i * np.arange(len(a))/len(a))
    a_tilde = a_sum/len(a)
    return a_tilde

def inv_DFT(a_tilde):
    a = 0
    for i in range(len(a_tilde)):
        a += a_tilde[i] * np.exp(-1j * 2*pi * i * np.arange(len(a_tilde))/len(a_tilde))
    return a

def beta(omega):
    beta_0 = 1
    beta_1 = 0
    beta_2 = 0
    beta_3 = 0
    omega_0 = 193.1e12/(2*pi)
    dw = omega - omega_0
    return beta_0 + dw*beta_1 + dw**2 * beta_2/2 + dw**3 * beta_3/6

dt = 1e-6 #time step
dz = 1e-3 #space step

N = 10 #Numberof space steps
M = 10 #number of time steps

beta_0 = 1
kappa = 0.1 #Kerr nonlinearity coefficient
beta_2 = 0.1 #second-order dispersion
omega_0 = 193.1e12/(2*pi) #1550 nm center frequency
A = np.zeros((M, N))
#A[:, 0] = np.ones(10)
for m in range(M): #initial gaussian pulse shape
    A[m, 0] = norm.pdf(m, 5)
print(A[:, 0])


#Compute Classical pulse envelope using split-step method

for n in range(N-1):
    if n%2 == 0:
        A[:, n+1] = A[:, n] * np.exp(1j * kappa * np.absolute(A[:, n]) * dz)
    else:
        A_tilde = np.zeros(M)
        for omega in range(M):
            A_tilde[omega] = DFT(A[:, n])[omega] * np.exp(1j * beta_2 * (omega - omega_0)**2 * dz)

        A[:, n+1] = inv_DFT(A_tilde)
     


mu = np.zeros((M, M, N))
mu[:, :, 0] = np.identity(M)
#print(mu[:, :, 0])

nu = np.zeros((M, M, N))
nu[:, :, 0] = np.identity(M)


for n in range(N-1):
    for k in range(M):
        mu[k, :, n+1] = ((1 + 2j*kappa*dz*(abs(A[k, n])**2) + 1j*beta_2*dz/(dt**2)) * mu[k, :, n]
                            + 1j*kappa*dz*(A[k, n])**2 * np.conjugate(nu[k, :, n])
                            + 1j * dz * inv_DFT((beta(2*pi*k/(M*dt) + omega_0) - beta_0) * DFT(mu[k, :, n]))) #- 1j * beta_2 * dz/(2*dt**2) * (mu[k-1, :, n] + mu[k+1, :, n]))

        nu[k, :, n+1] = ((1 + 2j*kappa*dz*(abs(A[k, n])**2) + 1j*beta_2*dz/(dt**2)) * nu[k, :, n]
                            + 1j*kappa*dz*(A[k, n])**2 * np.conjugate(nu[k, :, n])
                            + 1j * dz * inv_DFT((beta(2*pi*k/(M*dt) + omega_0) - beta_0) * DFT(nu[k, :, n]))) #- 1j * beta_2 * dz/(2*dt**2) * (nu[k-1, :, n] + nu[k+1, :, n]))
            

C = np.zeros(M)
C[:] = 2**0.5 * A[:, -1]


#Calculate maximum squeezing ratio
num = 0
denom = 0
for k in range(0, M):
    for l in range(0, M):
        for m in range(0, M):
            num += np.real(C[m] * np.conj(C[k]) * (np.conj(mu[m, l, -1]) * mu[k, l, -1] + np.conj(nu[m, l, -1]) * nu[k, l, -1])
                            + abs(np.conj(C[m]) * np.conj(C[k]) * (mu[m, l, -1] * nu[k, l, -1] + mu[k, l, -1] * nu[m, l, -1])))
    denom += abs(C[k])**2

print(num)
print(denom)

R_max = num/denom
print(R_max)
