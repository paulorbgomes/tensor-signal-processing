'''
Specific functions ...
'''

import numpy as np

def awgn_noise(signal_matrix,snr_db):
    noise = (1/np.sqrt(2)) * (np.random.normal(0,1,signal_matrix.shape) + 1j*np.random.normal(0,1,signal_matrix.shape))
    alpha = np.sqrt(((np.linalg.norm(signal_matrix,'fro')**2) / (np.linalg.norm(noise,'fro')**2)) * np.power(10,-snr_db/10))
    return signal_matrix + alpha*noise

def iid_channel(r,c):
    # r: rows, c: columns
    H = (1/np.sqrt(2)) * (np.random.normal(0,1,(r,c)) + 1j*np.random.normal(0,1,(r,c)))
    return H

def nmse(Ao, Ahat):
    return (np.linalg.norm(Ahat - Ao, 'fro')**2) / (np.linalg.norm(Ao, 'fro')**2)

def hadamard_product(A,B):
    (ra,ca) = A.shape
    (rb,cb) = B.shape
    if (ra == rb) and (ca == cb):
        C = np.zeros((ra,ca)) + 1j*np.zeros((ra,ca))
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
        C = np.zeros((ra*rb,ca)) + 1j*np.zeros((ra*rb,ca))
        for i in range(ca):
            C[:,i] = np.kron(A[:,i],B[:,i])
        return C
    else:
        print("Invalid operation!")

def ls_kraof(C,ra,rb):
    # C = khatri(A,B)
    (rc,cc) = C.shape
    A_hat = np.zeros((ra,cc)) + 1j*np.zeros((ra,cc))
    B_hat = np.zeros((rb,cc)) + 1j*np.zeros((rb,cc))
    for i in range(cc):
        ci = C[:,i].reshape(rb,ra,order='F') # rank-1 matrix
        U,S,Vh = np.linalg.svd(ci)
        B_hat[:,i] = np.sqrt(S[0]) * U[:,0]
        A_hat[:,i] = np.sqrt(S[0]) * Vh[0,:]
    return A_hat, B_hat

def ls_kronf(A,mb,nb):
    # A = kron(B,C)
    (m,n) = A.shape
    mc,nc = int(m/mb),int(n/nb)
    T = np.zeros((mc*nc,mb*nb)) + 1j*np.zeros((mc*nc,mb*nb))
    x = np.zeros((nb*nc,mb*nb)) + 1j*np.zeros((nb*nc,mb*nb))
    for iib in range(1,mb+1,1):
        for jjb in range(1,nb+1,1):
            x = A[(iib-1)*mc:iib*mc,(jjb-1)*nc:jjb*nc]
            vec_x = vec(x).squeeze()
            T[:,((jjb-1)*mb+iib)-1] = vec_x
    U,S,Vh = np.linalg.svd(T,full_matrices=False)
    C_hat = unvec(np.sqrt(S[0]) * U[:,0],mc,nc)
    B_hat = unvec(np.sqrt(S[0]) * Vh[0,:],mb,nb)
    return B_hat, C_hat
    
def vec(A):
    return A.reshape((A.size,1),order='F')

def unvec(a,ra,ca):
    return a.reshape((ra,ca),order='F')

def ten3_unfold(tenX,n):
    # Follows Kolda's notation ... 
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

        
















    
