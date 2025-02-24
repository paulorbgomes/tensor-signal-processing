'''
MIMO system example:
    M - tx antennas)
    N - rx antennas)
    T - symbol periods)
'''

# Y = HS + V

import numpy as np
import matplotlib.pyplot as plt

# Functions ...
def nmse(Ao, Ahat):
    return (np.linalg.norm(Ahat - Ao, 'fro')**2) / (np.linalg.norm(Ao, 'fro')**2)

# System model ...
M = 3
N = 5
T = 20

Ho = (1/np.sqrt(2)) * (np.random.normal(0,1,(N,M)) + 1j * np.random.normal(0,1,(N,M)))
So = np.random.choice([1,-1],(M,T))
Y = np.dot(Ho,So)

# Matched filter ...
Hhat = np.dot(Y,np.linalg.pinv(So))
Shat = np.dot(np.linalg.pinv(Ho),Y)

print("---------------------------")
print("NMSE metric: matched filter")
print(f"NMSE(H): {nmse(Ho,Hhat)}")
print(f"NMSE(S): {nmse(So,Shat)}")
print(f"NMSE(Y): {nmse(Y,np.dot(Hhat,Shat))}")
print("---------------------------")
print("")

# ALS ...
iterations = 100
v_error = []
Shat = np.random.choice([1,-1],(M,T))
for i in range(iterations):
    Hhat = np.dot(Y,np.linalg.pinv(Shat))
    Shat = np.dot(np.linalg.pinv(Hhat),Y)
    error = np.linalg.norm(Y - np.dot(Hhat,Shat),'fro')
    v_error.append(error)
    #print(error)

print("---------------------------")
print("NMSE metric: ALS")
print(f"NMSE(H): {nmse(Ho,Hhat)}")
print(f"NMSE(S): {nmse(So,Shat)}")
print(f"NMSE(Y): {nmse(Y,np.dot(Hhat,Shat))}")
print("---------------------------")
print("")

fig,ax = plt.subplots()
ax.plot(range(1,iterations+1),v_error,label="ALS")
ax.set_xlabel("ALS Iterations")
ax.set_ylabel("Reconstruction Error")
plt.legend()
plt.grid()
plt.show()
