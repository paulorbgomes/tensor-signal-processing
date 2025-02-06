'''
Homework 2 - Khatri-Rao Product
'''

import numpy as np
import scipy as sp
import scipy.linalg
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import functions as mf

# Problem 1 ...
I = [2,4,8,16,32,64,128,256]
R = [2,4]
monte_carlo = int(5e+3)

Final_1 = np.zeros((len(I),len(R)))
Final_2 = np.zeros((len(I),len(R)))
Final_3 = np.zeros((len(I),len(R)))

col = 0
for r in R:
    Method_1 = []
    Method_2 = []
    Method_3 = []
    for i in I:
        method_1 = []
        method_2 = []
        method_3 = []
        for j in range(monte_carlo):
            A = np.random.normal(0,1,(i,r))
            B = np.random.normal(0,1,(i,r))
            X = mf.khatri_rao_product(A,B)

            '''
            # Validation ...
            Xsci = sp.linalg.khatri_rao(A,B)
            print(mf.nmse(X,Xsci))

            pinvX = np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)
            M1 = np.linalg.pinv(mf.khatri_rao_product(A,B))
            print(mf.nmse(pinvX,M1))
    
            M2 = np.dot(np.linalg.inv(np.dot(mf.khatri_rao_product(A,B).T,mf.khatri_rao_product(A,B))),mf.khatri_rao_product(A,B).T)
            print(mf.nmse(pinvX,M2))

            M3 = np.dot(np.linalg.inv(np.multiply(np.dot(A.T,A),np.dot(B.T,B))),mf.khatri_rao_product(A,B).T)
            print(mf.nmse(pinvX,M3))
            '''

            start1 = timer()
            M1 = np.linalg.pinv(sp.linalg.khatri_rao(A,B))
            end1 = timer()
            method_1.append(end1 - start1)

            start2 = timer()
            M2 = np.dot(np.linalg.inv(np.dot(sp.linalg.khatri_rao(A,B).T,sp.linalg.khatri_rao(A,B))),sp.linalg.khatri_rao(A,B).T)
            end2 = timer()
            method_2.append(end2 - start2)

            start3 = timer()
            M3 = np.dot(np.linalg.inv(np.multiply(np.dot(A.T,A),np.dot(B.T,B))),sp.linalg.khatri_rao(A,B).T)
            end3 = timer()
            method_3.append(end3 - start3)

        Method_1.append(np.mean(method_1))
        Method_2.append(np.mean(method_2))
        Method_3.append(np.mean(method_3))

    Final_1[:,col] = np.array(Method_1)
    Final_2[:,col] = np.array(Method_2)
    Final_3[:,col] = np.array(Method_3)
    col += 1

# Plots ...
fig, ax = plt.subplots()
plt.yscale('log')
ax.plot(I,Final_1[:,0],marker="o",label="Method 1; R = 2")
ax.plot(I,Final_2[:,0],marker="o",label="Method 2; R = 2")
ax.plot(I,Final_3[:,0],marker="o",label="Method 3; R = 2")
ax.set_title("Problem 1")
ax.set_xlabel("I")
ax.set_ylabel("Processing Time (seconds)")
ax.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots()
plt.yscale('log')
ax.plot(I,Final_1[:,1],marker="o",label="Method 1; R = 4")
ax.plot(I,Final_2[:,1],marker="o",label="Method 2; R = 4")
ax.plot(I,Final_3[:,1],marker="o",label="Method 3; R = 4")
ax.set_title("Problem 1")
ax.set_xlabel("I")
ax.set_ylabel("Processing Time (seconds)")
ax.legend()
plt.grid()
plt.show()

print("")

# PRoblem 2 ...


















