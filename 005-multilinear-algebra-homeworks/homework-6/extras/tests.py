'''
Tests ...
'''

import functions as mf
import numpy as np
import scipy as sp
from scipy import linalg

'''
# 1. Hadamard Product ...
print("Hadamard Product ...")
A = (1/np.sqrt(2))*(np.random.normal(0,1,(3,3)) + 1j*np.random.normal(0,1,(3,3)))
B = (1/np.sqrt(2))*(np.random.normal(0,1,(3,3)) + 1j*np.random.normal(0,1,(3,3)))
C_mf = mf.hadamard_product(A,B)
C_np = np.multiply(A,B)
#print(C_mf - C_np)
print(f"NMSE: {mf.nmse(C_mf,C_np)}")
'''

'''
# 2. Khatri-Rao Product ...
A = (1/np.sqrt(2))*(np.random.normal(0,1,(3,3)) + 1j*np.random.normal(0,1,(3,3)))
B = (1/np.sqrt(2))*(np.random.normal(0,1,(2,3)) + 1j*np.random.normal(0,1,(2,3)))
C_mf = mf.khatri_rao_product(A,B)
C_sp = sp.linalg.khatri_rao(A,B)
#print(C_mf - C_sp)
print(f"NMSE: {mf.nmse(C_mf,C_sp)}")
'''

'''
# 3. LS-KRF ...
A = (1/np.sqrt(2))*(np.random.normal(0,1,(3,3)) + 1j*np.random.normal(0,1,(3,3)))
B = (1/np.sqrt(2))*(np.random.normal(0,1,(2,3)) + 1j*np.random.normal(0,1,(2,3)))
C_mf = mf.khatri_rao_product(A,B)
Ahat, Bhat = mf.ls_kraof(C_mf,3,2)

Chat = mf.khatri_rao_product(Ahat,Bhat)
print(f"NMSE_C: {mf.nmse(C_mf,Chat)}")

scA = Ahat/A
Ahat = Ahat@np.diag(1/scA[0,:])
print(f"NMSE_A: {mf.nmse(A,Ahat)}")

scB = Bhat/B
Bhat = Bhat@np.diag(1/scB[0,:])
print(f"NMSE_B: {mf.nmse(B,Bhat)}")
'''

'''
# 4. LS-KronF ...
A = (1/np.sqrt(2))*(np.random.normal(0,1,(3,3)) + 1j*np.random.normal(0,1,(3,3)))
B = (1/np.sqrt(2))*(np.random.normal(0,1,(2,2)) + 1j*np.random.normal(0,1,(2,2)))
C = np.kron(A,B)
Ahat,Bhat = mf.ls_kronf(C,3,3)

Chat = np.kron(Ahat,Bhat)
print(f"NMSE_C: {mf.nmse(C,Chat)}")

scA = Ahat[0,0]/A[0,0]
Ahat = (1/scA)*Ahat
print(f"NMSE_A: {mf.nmse(A,Ahat)}")

scB = Bhat[0,0]/B[0,0]
Bhat = (1/scB)*Bhat
print(f"NMSE_B: {mf.nmse(B,Bhat)}")
'''

'''
# 5. Vec / Unvec ...
A = (1/np.sqrt(2))*(np.random.normal(0,1,(3,3)) + 1j*np.random.normal(0,1,(3,3)))
vecA = A.reshape(np.prod(A.shape),-1,order="F")
vecA_mf = mf.vec(A)
print(f"NMSE_vec: {mf.nmse(vecA,vecA_mf)}")
unvecA = mf.unvec(vecA,A.shape[0],A.shape[1])
print(f"NMSE_unvec: {mf.nmse(A,unvecA)}")
'''

# 6. Unfold ...
I = 3
J = 5
K = 5
R = 4
A = np.random.normal(0,1,(I,R)) + 1j*np.random.normal(0,1,(I,R))
B = np.random.normal(0,1,(J,R)) + 1j*np.random.normal(0,1,(J,R))
C = np.random.normal(0,1,(K,R)) + 1j*np.random.normal(0,1,(K,R))
tenX = np.zeros((K,I,J)) + 1j*np.zeros((K,I,J))
for k in range(K):
    tenX[k,:,:] = A@np.diag(C[k,:])@(B.T)

X1 = mf.ten3_unfold(tenX,1)
X1_kolda = A@mf.khatri_rao_product(C,B).T
NMSE_X1 = mf.nmse(X1, X1_kolda)
print(f"NMSE(X1) = {NMSE_X1}")

X2 = mf.ten3_unfold(tenX,2)
X2_kolda = B@mf.khatri_rao_product(C,A).T
NMSE_X2 = mf.nmse(X2, X2_kolda)
print(f"NMSE(X2) = {NMSE_X2}")

X3 = mf.ten3_unfold(tenX,3)
X3_kolda = C@mf.khatri_rao_product(B,A).T
NMSE_X3 = mf.nmse(X3, X3_kolda)
print(f"NMSE(X3) = {NMSE_X3}")
    











