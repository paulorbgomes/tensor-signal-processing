'''
Homework 4: Least Squares Kronecker Product Factorization (LSKronF)
'''

import numpy as np
import functions as mf
import matplotlib.pyplot as plt

# Problem 1 ...
A = (1/np.sqrt(2))*(np.random.normal(0,1,(4,2)) + 1j*np.random.normal(0,1,(4,2)))
B = (1/np.sqrt(2))*(np.random.normal(0,1,(6,3)) + 1j*np.random.normal(0,1,(6,3)))
X = np.kron(A,B)

A_hat,B_hat = mf.ls_kronf(X,4,2)
scaA = A_hat/A
A_hat = (1/scaA) * A_hat
NMSE_A = mf.nmse(A,A_hat)
scaB = B_hat/B
B_hat = (1/scaB) * B_hat
NMSE_B = mf.nmse(B,B_hat)

print("Problem 1 ...")
print(f"NMSE_A = {NMSE_A}")
print(f"NMSE_B = {NMSE_B}")
print("")

# Problem 2 ...
I = [2,4]
J = [4,8]
P = [3,3]
Q = [5,5]

SNR = range(0,35,5)
monte_carlo = int(1e+3)

Final_X = np.zeros((len(SNR),len(I)))
col = 0

for i in range(len(I)):
    NMSE_X = []
    for snr in SNR:
        nmse_X = []
        for j in range(monte_carlo):
            A = (1/np.sqrt(2)) * (np.random.normal(0,1,(I[i],P[i])) + 1j * np.random.normal(0,1,(I[i],P[i])))
            B = (1/np.sqrt(2)) * (np.random.normal(0,1,(J[i],Q[i])) + 1j * np.random.normal(0,1,(J[i],Q[i])))
            Xo = np.kron(A,B)
            X = mf.awgn_noise(Xo,snr)

            '''
            noise = X - Xo
            SNRcalc = 10 * np.log10((np.linalg.norm(Xo,'fro')**2) / (np.linalg.norm(noise,'fro')**2))
            print(SNRcalc)
            '''
            
            A_hat,B_hat = mf.ls_kronf(X,I[i],P[i])
            '''
            scaA = A_hat/A
            A_hat = np.dot(A_hat,np.linalg.inv(np.diag(scaA[1,:])))
            scaB = B_hat/B
            B_hat = np.dot(B_hat,np.linalg.inv(np.diag(scaB[1,:])))
            '''
            X_hat = np.kron(A_hat,B_hat) 
            
            nmse_X.append(mf.nmse(Xo,X_hat))

        NMSE_X.append(np.mean(nmse_X))

    Final_X[:,col] = np.array(NMSE_X)
    col += 1

# Plots ...
fig, ax = plt.subplots()
plt.yscale("log")
ax.plot(SNR,Final_X[:,0],marker="o",label="I = 2 , J = 4 , P = 3, Q = 5")
ax.plot(SNR,Final_X[:,1],marker="o",label="I = 4 , J = 8 , P = 3, Q = 5")
ax.set_title("Problem 2")
ax.set_xlabel("SNR(dB)")
ax.set_ylabel("NMSE(X)")
plt.legend()
plt.grid()
plt.show()


    
    

    
    
    
   
    

            

    



    
    

    


    

