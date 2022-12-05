#import scipy.io
import math
from math import pi, sqrt, exp
from mpmath import sech
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import matplotlib as mlp
from matplotlib.animation import FuncAnimation
import random
from optparse import OptionParser
import csv
import os
from scipy.stats import norm


beta_0 = 2 * pi * 1.467 / (1.55e-9) # 1/km
beta_1 = 0 #ps/km
beta_2 = -20 #ps^2/km
beta_3 = 0.1 #ps^3/km
kappa = 1.467 #W^-1 km^-1  Kerr nonlinearity coefficient
omega_0 = 193.1e12/(2*pi) #1550 nm center frequency

def beta(omega):
    dw = omega - omega_0
    return beta_0 + dw*beta_1 + dw**2 * beta_2/2 + dw**3 * beta_3/6


c_fiber = 2.99792e-7/1.467 #km/ps

z_max = 0.000001 #km
t_max = z_max/c_fiber #ps

M = 1000 #number of time steps
N = 1000 #Number of space steps

dt =  t_max/M  #1e-6 #ps  time step
print('dt: ', dt)
dz = z_max/N #km  space step


beta_0 = 2 * pi * 1.467 / (1.55e-9)
kappa = 1.5 #W^-1 km^-1  Kerr nonlinearity coefficient
beta_2 = -20  #ps^2/km   second-order dispersion
omega_0 = 193.1e12/(2*pi) #1550 nm center frequency
FWHM = 0.2 #ps  FWHM 


A = np.zeros((M, N), dtype = 'complex_')
#A[:, 0] = np.ones(10)
for m in range(M): #initial gaussian pulse shape
    A[m, 0] = 1e6 * norm.pdf(dt*m, 0.5, FWHM/2.355)

plt.plot(A[:, 0])
plt.show()


#print(A[:, 0])


#Compute classical pulse envelope using split-step method

for n in range(N-1):
        A_NL = A[:, n] * np.exp(1j * kappa * np.absolute(A[:, n])**2 * dz/2) #Nonlinear step
        #print(np.exp(1j * kappa * np.absolute(A[:, n])**2 * dz))

        A_tilde = fft(A_NL)
        A_D = np.zeros(M, dtype = 'complex_')
        for f in range(M): #Dispersion step in frequency space
            A_D[f] = A_tilde[f] * np.exp(1j * beta_2 * (2*pi*f/dt - omega_0)**2 * dz/2)

        A[:, n+1] = ifft(A_D)
     

"""
mu = np.zeros((M, M, N), dtype = 'complex_')
mu[:, :, 0] = np.identity(M ,dtype = 'complex_')
#print(mu[:, :, 0])

nu = np.zeros((M, M, N), dtype = 'complex_')
nu[:, :, 0] = np.identity(M, dtype = 'complex_')


for n in range(N-1):
    if n%50 == 0:
            print(n, " / ", N)
    for k in range(M):
        mu[k, :, n+1] = ((1 + 2j*kappa*dz*(abs(A[k, n])**2) + 1j*beta_2*dz/(dt**2)) * mu[k, :, n]
                            + 1j*kappa*dz*(A[k, n])**2 * np.conjugate(nu[k, :, n])
                            + 1j * dz * ifft((beta(2*pi*k/(M*dt) + omega_0) - beta_0) * fft(mu[k, :, n]))) #- 1j * beta_2 * dz/(2*dt**2) * (mu[k-1, :, n] + mu[k+1, :, n]))

        nu[k, :, n+1] = ((1 + 2j*kappa*dz*(abs(A[k, n])**2) + 1j*beta_2*dz/(dt**2)) * nu[k, :, n]
                            + 1j*kappa*dz*(A[k, n])**2 * np.conjugate(nu[k, :, n])
                            + 1j * dz * ifft((beta(2*pi*k/(M*dt) + omega_0) - beta_0) * fft(nu[k, :, n]))) #- 1j * beta_2 * dz/(2*dt**2) * (nu[k-1, :, n] + nu[k+1, :, n]))
            

C = np.zeros(M, dtype = 'complex_')
C[:] = 2**0.5 * A[:, -1]
"""
"""
#Calculate maximum squeezing ratio
num = 0
denom = 0
for k in range(0, M):
    if k%10 == 0:
            print(k, " / ", M)
    for l in range(0, M):
        for m in range(0, M):
            num += np.real(C[m] * np.conj(C[k]) * (np.conj(mu[m, l, -1]) * mu[k, l, -1] + np.conj(nu[m, l, -1]) * nu[k, l, -1])
                            + abs(np.conj(C[m]) * np.conj(C[k]) * (mu[m, l, -1] * nu[k, l, -1] + mu[k, l, -1] * nu[m, l, -1])))
    denom += abs(C[k])**2

print(num)
print(denom)

R_max = num/denom
print(R_max)
"""
"""
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)

ax[0].title.set_text('t = 0')
ax[1].title.set_text('t = end')

ax[0].plot(np.linspace(0, z_max, N), A[0, :])
ax[1].plot(np.linspace(0, z_max, N), A[1, :])
plt.show()
"""

fig = plt.figure()
ax = plt.axes(xlim=(0, z_max), ylim=(-A.max(), A.max()))
line, = ax.plot([], [], lw=3)
ax.set_xlabel('Distance [km]')
ax.set_ylabel('Classical Pulse Envelope [W^0.5 * km^-1]')


def init():
    line.set_data([], [])
    return line,
def animate(i):
    x = np.linspace(0, z_max, N)
    y = A[i, :]
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, animate, init_func=init, frames=M, interval=100, blit=True)
plt.show()
