'''
Specific functions ...
'''

import numpy as np

def awgn_noise(signal_matrix,snr_db):
    noise = (1/np.sqrt(2)) * (np.random.normal(0,1,signal_matrix.shape) + 1j*np.random.normal(0,1,signal_matrix.shape))
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

def ls_kraof(C,ra,rb):
    (rc,cc) = C.shape
    A_hat = np.zeros((ra,cc)) + 1j*np.zeros((ra,cc))
    B_hat = np.zeros((rb,cc)) + 1j*np.zeros((rb,cc))
    for i in range(cc):
        ci = C[:,i].reshape(rb,ra,order='F') # rank-1 matrix
        U,S,Vh = np.linalg.svd(ci)
        B_hat[:,i] = np.sqrt(S[0]) * U[:,0]
        A_hat[:,i] = np.sqrt(S[0]) * Vh[0,:]
    return A_hat, B_hat

def vec(A):
    return A.reshape((A.size,1),order='F')

def unvec(a,ra,ca):
    return a.reshape((ra,ca),order='F')
