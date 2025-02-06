'''
Homework 1 - Hadamard, Kronecker and Khatri-Rao Products
'''

import scipy as sp
import scipy.linalg
import numpy as np
import functions as mf
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Problem 1 ...
N = [2,4,8,16,32,64,128]
monte_carlo = int(5e+3)

Method_1 = []
Method_2 = []
for n in N:
    print(f"N = {n}")
    method_1 = []
    method_2 = []
    for i in range(monte_carlo):
        A = np.random.normal(0,1,(n,n))
        B = np.random.normal(0,1,(n,n))
        
        start1 = timer()
        C = mf.hadamard_product(A,B)
        end1 = timer()
        method_1.append(end1 - start1)

        start2 = timer()
        D = np.multiply(A,B)
        end2 = timer()
        method_2.append(end2 - start2)
    Method_1.append(np.mean(method_1))
    Method_2.append(np.mean(method_2))
print(mf.nmse(C,D))

# Plots ...
fig, ax = plt.subplots()
plt.yscale('log')
ax.plot(N,Method_1,marker='o',label="Proposed method")
ax.legend()
ax.plot(N,Method_2,marker='o',label="Numpy method")
ax.legend()
ax.set_title("Problem 1")
ax.set_xlabel("N")
ax.set_ylabel("Processing Time (seconds)")
plt.grid()
plt.show()

print("")

# Problem 2 ...

# Problem 3 ...
Method_1 = []
Method_2 = []
for n in N:
    print(f"N = {n}")
    method_1 = []
    method_2 = []
    for i in range(monte_carlo):
        A = np.random.normal(0,1,(n,n))
        B = np.random.normal(0,1,(n,n))
        
        start3 = timer()
        C = mf.khatri_rao_product(A,B)
        end3 = timer()
        method_1.append(end3 - start3)
     
        start4 = timer()
        D = sp.linalg.khatri_rao(A,B)
        end4 = timer()
        method_2.append(end4 - start4)
    Method_1.append(np.mean(method_1))
    Method_2.append(np.mean(method_2))
print(mf.nmse(C,D))

# Plots ...
fig, ax = plt.subplots()
ax.plot(N,Method_1,marker='o',label="Proposed method")
ax.plot(N,Method_2,marker='o',label="Scipy method")
ax.set_yscale('log')
ax.set_title("Problem 3")
ax.set_xlabel("N")
ax.set_ylabel("Processing Time (seconds)")
ax.legend()
ax.grid(True)
plt.show()








        









    
    


    












