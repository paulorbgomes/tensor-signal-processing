'''
Homework 9: Multidimensional Least-Squares Khatri-Rao Factorization (MLS-KRF)
'''

import numpy as np
import functions as mf
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg

# Problem 1 ...
I1 = 3
I2 = 4
I3 = 5
R = 3

A1 = (1/np.sqrt(2)) * (np.random.normal(0,1,(I1,R)) + 1j*np.random.normal(0,1,(I1,R)))
A2 = (1/np.sqrt(2)) * (np.random.normal(0,1,(I2,R)) + 1j*np.random.normal(0,1,(I2,R)))
A3 = (1/np.sqrt(2)) * (np.random.normal(0,1,(I3,R)) + 1j*np.random.normal(0,1,(I3,R)))
X = mf.khatri_rao_product(mf.khatri_rao_product(A1,A2),A3)
A1_hat, A2_hat, A3_hat = mf.mls_kraof(X,I1,I2,I3)
X_hat = mf.khatri_rao_product(mf.khatri_rao_product(A1_hat,A2_hat),A3_hat)

NMSE_X = mf.nmse(X,X_hat)

scaA1 = A1_hat/A1
A1_hat = np.dot(A1_hat,np.linalg.inv(np.diag(scaA1[0,:])))
NMSE_A1 = mf.nmse(A1,A1_hat)

scaA2 = A2_hat/A2
A2_hat = np.dot(A2_hat,np.linalg.inv(np.diag(scaA2[0,:])))
NMSE_A2 = mf.nmse(A2,A2_hat)

scaA3 = A3_hat/A3
A3_hat = np.dot(A3_hat,np.linalg.inv(np.diag(scaA3[0,:])))
NMSE_A3 = mf.nmse(A3,A3_hat)

print("Problem 1 ...")
print(f"NMSE_X = {NMSE_X}")
print(f"NMSE_A1 = {NMSE_A1}")
print(f"NMSE_A2 = {NMSE_A2}")
print(f"NMSE_A3 = {NMSE_A3}")
print("")

# Problem 2 ...
I = 2
J = 3
K = 4
R = 5

SNR = range(0,35,5)
monte_carlo = int(1e+3)

NMSE = []
for snr in SNR:
    nmse = []
    for j in range(monte_carlo):
        A = (1/np.sqrt(2)) * (np.random.normal(0,1,(I,R)) + 1j*np.random.normal(0,1,(I,R)))
        B = (1/np.sqrt(2)) * (np.random.normal(0,1,(J,R)) + 1j*np.random.normal(0,1,(J,R)))
        C = (1/np.sqrt(2)) * (np.random.normal(0,1,(K,R)) + 1j*np.random.normal(0,1,(K,R)))
        Xo = mf.khatri_rao_product(mf.khatri_rao_product(A,B),C)
        X = mf.awgn_noise(Xo,snr) 

        A_hat,B_hat,C_hat = mf.mls_kraof(X,I,J,K)
        X_hat = mf.khatri_rao_product(mf.khatri_rao_product(A_hat,B_hat),C_hat)
        nmse.append(mf.nmse(Xo,X_hat))
        
    NMSE.append(np.mean(nmse))

# Plots ...
fig, ax = plt.subplots()
plt.yscale("log")
ax.plot(SNR,NMSE,marker="o")
ax.set_title("Problem 2")
ax.set_xlabel("SNR(dB)")
ax.set_ylabel("NMSE(X)")
plt.grid()
plt.show()
