'''
Homework 3: Least-Squares Khatri-Rao Factorization (LSKRF)
'''

import numpy as np
import functions as mf
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg

# Problem 1 ...
A = np.random.normal(0,1,(4,2)) + 1j * np.random.normal(0,1,(4,2))
B = np.random.normal(0,1,(6,2)) + 1j * np.random.normal(0,1,(6,2))
C = sp.linalg.khatri_rao(A,B)

A_hat,B_hat = mf.ls_kraof(C,4,6)
scaA = A_hat/A
A_hat = np.dot(A_hat,np.linalg.inv(np.diag(scaA[1,:])))
NMSE_A = mf.nmse(A,A_hat)

scaB = B_hat/B
B_hat = np.dot(B_hat,np.linalg.inv(np.diag(scaB[1,:])))
NMSE_B = mf.nmse(B,B_hat)

print("Problem 1 ...")
print(f"NMSE_A = {NMSE_A}")
print(f"NMSE_B = {NMSE_B}")
print("")

# Problem 2 ...
I = [10,30]
J = [10,10]
R = 4

SNR = range(0,35,5)
monte_carlo = int(1e+2)

Final_X = np.zeros((len(SNR),len(I)))
col = 0

for i in range(len(I)):
    NMSE_X = []
    for snr in SNR:
        nmse_X = []
        for j in range(monte_carlo):
            A = np.random.normal(0,1,(I[i],R)) + 1j * np.random.normal(0,1,(I[i],R))
            B = np.random.normal(0,1,(J[i],R)) + 1j * np.random.normal(0,1,(J[i],R))
            Xo = sp.linalg.khatri_rao(A,B)
            X = mf.awgn_noise(Xo,snr) 

            A_hat,B_hat = mf.ls_kraof(X,I[i],J[i])
            X_hat = sp.linalg.khatri_rao(A_hat,B_hat) 
            #scaA = A_hat/A
            #A_hat = np.dot(A_hat,np.linalg.inv(np.diag(scaA[1,:])))
            #scaB = B_hat/B
            #B_hat = np.dot(B_hat,np.linalg.inv(np.diag(scaB[1,:])))
               
            nmse_X.append(mf.nmse(Xo,X_hat))

        NMSE_X.append(np.mean(nmse_X))

    Final_X[:,col] = np.array(NMSE_X)
    col += 1

# Plots ...
fig, ax = plt.subplots()
plt.yscale("log")
ax.plot(SNR,Final_X[:,0],marker="o",label="I = 10 , J = 10 , R = 4")
ax.plot(SNR,Final_X[:,1],marker="o",label="I = 30 , J = 10 , R = 4")
ax.set_title("Problem 2")
ax.set_xlabel("SNR(dB)")
ax.set_ylabel("NMSE(X)")
plt.legend()
plt.grid()
plt.show()






















