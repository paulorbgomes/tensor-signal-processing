'''
Homework 0 - Kronecker Product
'''

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import functions as mf

# Problem 1 ...
# Item a ...
N = [2,4,8,16,32,64]
Method_1 = []
Method_2 = []
monte_carlo = int(1e+2)

for n in N:
    print(f"N = {n}")
    method_1 = []
    method_2 = []
    for i in range(monte_carlo):  
        A = np.random.normal(0,1,(n,n))
        B = np.random.normal(0,1,(n,n))

        # Method 1 ...
        start = timer()
        C1 = np.linalg.inv(np.kron(A,B))
        end = timer()
        method_1.append(end - start)

        # Method 2 ...
        start = timer()
        C2 = np.kron(np.linalg.inv(A),np.linalg.inv(B))
        end = timer()
        method_2.append(end - start)
        
    Method_1.append(np.mean(method_1))
    Method_2.append(np.mean(method_2))
#print(f"NMSE = {f.nmse(C1,C2)}")

# Plots ...
fig, ax = plt.subplots()
plt.yscale('log')
ax.plot(N,Method_1,marker='o',label="Method 1")
ax.legend()
ax.plot(N,Method_2,marker='*',label="Method 2")
ax.legend()
ax.set_title("Problem 1")
ax.set_xlabel("N")
ax.set_ylabel("Processing Time (seconds)")
plt.grid()
plt.show()

print("")

# Item b ...
K = [2,4,6,8,10]
N = 2
Method_1 = []
# Method 1 ...
for i in range(0,len(K)):
    print(f"K = {K[i]}")
    method_1 = []
    for j in range(monte_carlo):
        # Method 1 ...
        C1 = np.random.normal(0,1,(N,N))
        time1 = 0
        for q in range(1,K[i]):
            start1 = timer()
            C1 = np.kron(C1,np.random.normal(0,1,(N,N)))
            end1 = timer()
            time1 = time1 + (end1 - start1)
        #print(C1.shape)
        start2 = timer()
        C1_inv = np.linalg.inv(C1)
        end2 = timer()
        time2 = end2 - start2
        time_f = time1 + time2
        method_1.append(time_f)
    Method_1.append(np.mean(method_1))

# Method 2 ...
Method_2 = []
for i in range(0,len(K)):
    method_2 = []
    for j in range(monte_carlo):
        C2 = np.random.normal(0,1,(N,N)) 
        start1 = timer()
        C2 = np.linalg.inv(C2)
        end1 = timer()
        time1 = end1 - start1
        time2 = 0
        for q in range(1,K[i]):
            I = np.random.normal(0,1,(N,N))
            start2 = timer()
            I = np.linalg.inv(I)
            C2 = np.kron(C2,I)
            end2 = timer()
            time2 = time2 + (end2 - start2)
        time_f = time1 + time2
        method_2.append(time_f)
    Method_2.append(np.mean(method_2))

# Plots ...
fig, ax = plt.subplots()
plt.yscale('log')
ax.plot(K,Method_1,marker='o',label="Method 1")
ax.plot(K,Method_2,marker='*',label="Method 2")
ax.legend()
ax.set_title("Problem 2")
ax.set_xlabel("K")
ax.set_ylabel("Processing Time (seconds)")
plt.grid()
plt.show()
