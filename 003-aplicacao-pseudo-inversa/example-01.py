'''
MIMO system example:
    M - tx antennas)
    N - rx antennas)
    T - symbol periods)
'''

import numpy as np

# Functions ...
def nmse(Ao, Ahat):
    return (np.linalg.norm(Ahat - Ao, 'fro')**2) / (np.linalg.norm(Ao, 'fro')**2)

# System model ...
M = 3
N = 5
T = 10

Ho = np.random.normal(0,1,(N,M)) + 1j * np.random.normal(0,1,(N,M))
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
Shat = np.random.choice([1,-1],(M,T))
for i in range(100):
    Hhat = np.dot(Y,np.linalg.pinv(Shat))
    Shat = np.dot(np.linalg.pinv(Hhat),Y)
    error = np.linalg.norm(Y - np.dot(Hhat,Shat),'fro')
    #print(error)

print("---------------------------")
print("NMSE metric: ALS")
print(f"NMSE(H): {nmse(Ho,Hhat)}")
print(f"NMSE(S): {nmse(So,Shat)}")
print(f"NMSE(Y): {nmse(Y,np.dot(Hhat,Shat))}")
print("---------------------------")
print("")

    











