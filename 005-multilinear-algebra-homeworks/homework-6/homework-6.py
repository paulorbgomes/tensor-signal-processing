'''
Homework 6: Unfolding, Folding and n-Mode Product
'''

import numpy as np
import functions as mf

'''
def ten3_fold(tenX_unf,v_dims,n):
    # v_dims = [I,J,K]
    I = v_dims[0]
    J = v_dims[1]
    K = v_dims[2]
    tenX = np.zeros((K,I,J)) + 1j*np.zeros((K,I,J))
    if n == 1:
        for k in range(K):
            kc = k + 1
            tenX[k,:,:] = tenX_unf[:,(kc-1)*J:(kc*J)]
    elif n == 2:
        for k in range(K):
            kc = k + 1
            tenX[k,:,:] = tenX_unf[:,(kc-1)*I:(kc*I)].T
    elif n == 3:
        for j in range(J):
            jc = j + 1
            tenX[:,:,j] = tenX_unf[:,(jc-1)*I:(jc*I)]
    return tenX
    

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
'''

# Problem 1 ...
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

X1 = mf.ten3_unfold(tenX,1)
X2 = mf.ten3_unfold(tenX,2)
X3 = mf.ten3_unfold(tenX,3)

print(f"{X1.shape}")
print(f"{X1} \n")
print(f"{X2.shape}")
print(f"{X2} \n")
print(f"{X3.shape}")
print(f"{X3} \n")

# Problem 2 ...
X1 = mf.ten3_unfold(tenX,1)
tenX_fold = mf.ten3_fold(X1,[3,4,2],1)
NMSE_mode1 = mf.nmse(mf.ten3_unfold(tenX,1), mf.ten3_unfold(tenX_fold,1))
print(f"NMSE(mode_1) = {NMSE_mode1} ... Tensor-Shape: {tenX_fold.shape}")

X2 = mf.ten3_unfold(tenX,2)
tenX_fold = mf.ten3_fold(X2,[3,4,2],2)
NMSE_mode2 = mf.nmse(mf.ten3_unfold(tenX,2), mf.ten3_unfold(tenX_fold,2))
print(f"NMSE(mode_2) = {NMSE_mode2} ... Tensor-Shape: {tenX_fold.shape}")

X3 = mf.ten3_unfold(tenX,3)
tenX_fold = mf.ten3_fold(X3,[3,4,2],3)
NMSE_mode3 = mf.nmse(mf.ten3_unfold(tenX,3), mf.ten3_unfold(tenX_fold,3))
print(f"NMSE(mode_3) = {NMSE_mode3} ... Tensor-Shape: {tenX_fold.shape}")


# Problem 3 ...
I = 2
J = 3
K = 4
R = 5

A = (1/np.sqrt(2))*(np.random.normal(0,1,(I,R)) + 1j*np.random.normal(0,1,(I,R)))
B = (1/np.sqrt(2))*(np.random.normal(0,1,(J,R)) + 1j*np.random.normal(0,1,(J,R)))
C = (1/np.sqrt(2))*(np.random.normal(0,1,(K,R)) + 1j*np.random.normal(0,1,(K,R)))

tenX_1 = mf.ten3_parafac(A,B,C)
tenX_2 = mf.ten3_nmode_product(mf.ten3_eye(R),A,1)
tenX_2 = mf.ten3_nmode_product(tenX_2,B,2)
tenX_2 = mf.ten3_nmode_product(tenX_2,C,3)

NMSE = mf.nmse(mf.ten3_unfold(tenX_1,1), mf.ten3_unfold(tenX_2,1))
print(f"NMSE = {NMSE}")
