'''
Homework 8: High Order Orthogonal Iteration (HOOI)
'''

import numpy as np
import functions as mf
import matplotlib.pyplot as plt

# Problem 1 ...
I = 5
J = 6
K = 7
Ig = 3
Jg = 4
Kg = 5

# Factor matrices ...
A = (1/np.sqrt(2)) * (np.random.normal(0,1,(I,Ig)) + 1j*np.random.normal(0,1,(I,Ig)))
B = (1/np.sqrt(2)) * (np.random.normal(0,1,(J,Jg)) + 1j*np.random.normal(0,1,(J,Jg)))
C = (1/np.sqrt(2)) * (np.random.normal(0,1,(K,Kg)) + 1j*np.random.normal(0,1,(K,Kg)))
G1 = (1/np.sqrt(2)) * (np.random.normal(0,1,(Ig,Jg*Kg)) + 1j*np.random.normal(0,1,(Ig,Jg*Kg)))
tenG = mf.ten3_fold(G1,[Ig,Jg,Kg],1)
tenX = mf.ten3_tucker3(tenG,A,B,C)
tenX_hosvd = mf.ten3_hosvd(tenX,Ig,Jg,Kg)
tenX_hooi = mf.ten3_hooi(tenX,Ig,Jg,Kg,1e+3)
NMSE_hosvd = mf.nmse(mf.ten3_unfold(tenX,1),mf.ten3_unfold(tenX_hosvd,1))
NMSE_hooi = mf.nmse(mf.ten3_unfold(tenX,1),mf.ten3_unfold(tenX_hooi,1))
print("Problem 1 ...")
print(f"NMSE_HOSVD = {NMSE_hosvd}")
print(f"NMSE_HOOI = {NMSE_hooi}")

# Problem 2 ...
I = 10
R = 10
A = (1/np.sqrt(2)) * (np.random.normal(0,1,(I,R)) + 1j*np.random.normal(0,1,(I,R)))
B = (1/np.sqrt(2)) * (np.random.normal(0,1,(I,R)) + 1j*np.random.normal(0,1,(I,R)))
C = (1/np.sqrt(2)) * (np.random.normal(0,1,(I,R)) + 1j*np.random.normal(0,1,(I,R)))
G1 = (1/np.sqrt(2)) * (np.random.normal(0,1,(R,R*R)) + 1j*np.random.normal(0,1,(R,R*R)))
tenG = mf.ten3_fold(G1,[R,R,R],1)
tenX = mf.ten3_tucker3(tenG,A,B,C)

error = np.zeros(R)
for r in range(R):
    i = r + 1
    tenX_hat = mf.ten3_hooi(tenX,i,i,i,1e+3)
    error[r] = mf.nmse(mf.ten3_unfold(tenX,1),mf.ten3_unfold(tenX_hat,1))
fig,ax = plt.subplots()
plt.yscale("log")
ax.plot(range(1,11,1), error, marker="o", linewidth=3)
ax.set_title("HOOI Low-Rank Approximation", fontsize=14)
ax.set_xlabel("R", fontsize=12)
ax.set_ylabel("NMSE", fontsize=12)
plt.xlim((1,10))
plt.grid()
plt.show() 
