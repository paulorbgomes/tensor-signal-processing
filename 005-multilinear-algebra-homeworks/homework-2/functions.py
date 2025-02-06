'''
Specific functions ...
'''
import numpy as np

def awgn_noise(signal_matrix,snr_db):
    noise = np.random.normal(0,1,signal_matrix.shape)
    alpha = np.sqrt(((np.linalg.norm(signal_matrix,'fro')**2) / (np.linalg.norm(noise,'fro')**2)) * np.power(10,-snr_db/10))
    return signal_matrix + alpha*noise

def nmse(Ao, Ahat):
    return (np.linalg.norm(Ahat - Ao, 'fro')**2) / (np.linalg.norm(Ao, 'fro')**2)

def hadamard_product(A,B):
    (ra,ca) = A.shape
    (rb,cb) = B.shape
    if (ra == rb) and (ca == cb):
        C = np.zeros((ra,ca))
        for i in range(ra):
            for j in range(ca):
                C[i,j] = A[i,j]*B[i,j]
        return C
    else:
        print("Invalid operation!")

def khatri_rao_product(A,B):
    (ra,ca) = A.shape
    (rb,cb) = B.shape
    if (ca == cb):
        C = np.zeros((ra*rb,ca))
        for i in range(ca):
            C[:,i] = np.kron(A[:,i],B[:,i])
        return C
    else:
        print("Invalid operation!")
        
















    
