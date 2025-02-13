'''
Homework 6: Unfolding, Folding and n-Mode Product
'''

import numpy as np
import functions as mf

def ten3_unfold(tenX,n):
    # Follow Kolda's notation ... 
    K,I,J = tenX.shape
    if n == 1: # mode-1 unfolding
        X1 = np.zeros((I,J*K)) + 1j*np.zeros((I,J*K))
        col = 0
        for k in range(K):
            for j in range(J):
                X1[:,col] = tenX[k,:,j]
                col += 1
        return X1
    elif n == 2: # mode-2 unfolding
        X2 = np.zeros((J,I*K)) + 1j*np.zeros((J,I*K))
        col = 0
        for k in range(K):
            for i in range(I):
                X2[:,col] = tenX[k,i,:]
                col += 1
        return X2
    elif n == 3: # mode-3 unfolding
        X3 = np.zeros((K,I*J)) + 1j*np.zeros((K,I*J))
        col = 0
        for j in range(J):
            for i in range(I):
                X3[:,col] = tenX[:,i,j]
                col += 1
        return X3

# Kolda's example ...
tenX = np.zeros((2,3,4)) + 1j*np.zeros((2,3,4))
tenX[0,:,:] = np.array([
                        [1,4,7,10],
                        [2,5,8,11],
                        [3,6,9,12]
                       ])

tenX[1,:,:] = np.array([
                        [13,16,19,22],
                        [14,17,20,23],
                        [15,18,21,24]
                       ])
print(f"{tenX.shape} \n")

X1 = ten3_unfold(tenX,1)
X2 = ten3_unfold(tenX,2)
X3 = ten3_unfold(tenX,3)

print(f"{X1.shape}")
print(f"{X1} \n")
print(f"{X2.shape}")
print(f"{X2} \n")
print(f"{X3.shape}")
print(f"{X3} \n")
