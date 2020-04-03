# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:42:52 2020

@author: yevgeniy simonov
"""
import numpy as np
import matplotlib.pyplot as plt
from skrf import Network #used for reading .snp files
from numpy.fft import ifft

if __name__ == "__main__":
    
    #paths to S-parameter data
    file_300 = 'BBHA9120D_300.s1p'
    file_1201 = 'BBHA9120D_1201.s1p'
    
    #frequencies in Hz
    S11_freq = np.squeeze(Network(file_300).frequency.f)

    #S11 parameters of Serial 300
    S11_300 = np.squeeze(Network(file_300).s)
    
    #S11 parameters of Serial 1201
    S11_1201 = np.squeeze(Network(file_1201).s)
    
    #defermine sampling frequency, assuming Nyquist Rate
    fs = 2.0 * np.amax(S11_freq)
    
    #determine the number of points in IFFT
    N = fs / (S11_freq[1] - S11_freq[0]) #2 ** 20
    
    #find the closest location, so that 2^x1 < N < 2^x2
    for i in range(1,20):
        x = 2 ** i
        if(x >= N):
            N = x
            break
    
    #compute inverse Fourier transforms of the data 
    s11_300t = ifft(S11_300, N)
    s11_1201t = ifft(S11_1201, N)
    
    #figure out what the time axis is
    dt = 1.0 / fs
    tmax = N * dt
    t = np.arange(0, tmax, dt) * 1e9

    #plot results
    fig, a = plt.subplots(2,2,sharey=True)
    a[0][0].plot(t, s11_300t.real, color = 'blue')
    a[0][1].plot(t, s11_300t.imag, color = 'red')
    a[1][0].plot(t, s11_1201t.real, color = 'blue')
    a[1][1].plot(t, s11_1201t.imag, color = 'red')
    
    for i in range(0,2):
        for j in range(0,2):
            a[i][j].set_yscale('log')
            a[i][j].grid()
            a[i][j].set_xlabel(r'time, ns')

    a[0][0].set_title('Re(s11), 300 (faulty)', fontsize=10)
    a[0][1].set_title('Im(s11), 300 (faulty)', fontsize=10)
    a[1][0].set_title('Re(s11), 1201', fontsize=10)
    a[1][1].set_title('Im(s11), 1201', fontsize=10)
    
    a[0][0].set_ylabel(r'$\log_{10}\Re\{\mathfrak{F}^{-1}[S11]\}$')
    a[0][1].set_ylabel(r'$\log_{10}\Im\{\mathfrak{F}^{-1}[S11]\}$')
    a[1][0].set_ylabel(r'$\log_{10}\Re\{\mathfrak{F}^{-1}[S11]\}$')
    a[1][1].set_ylabel(r'$\log_{10}\Im\{\mathfrak{F}^{-1}[S11]\}$')
    
    fig.tight_layout(pad=1.0)
    plt.savefig(str(N)+'-point_IFFT.png', dpi=1000)

    
    