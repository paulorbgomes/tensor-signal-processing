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

def ten3_fold(tenX_unf,v_dims,n):
    # v_dims = [I,J,K] (vector with dimensions of the third order tensor)
    I = v_dims[0]
    J = v_dims[1]
    K = v_dims[2]
    tenX = np.zeros((K,I,J)) + 1j*np.zeros((K,I,J))
    if n == 1:
        for k in range(K):
            tenX[k,:,:] = tenX_unf[:,k*J:(k+1)*J]
    elif n == 2:
        for k in range(K):
            tenX[k,:,:] = tenX_unf[:,k*I:(k+1)*I].T
    elif n == 3:
        for j in range(J):
            tenX[:,:,j] = tenX_unf[:,j*I:(j+1)*I]
    return tenX

def ten3_eye(N):
    tenI = np.zeros((N,N,N)) + 1j*np.zeros((N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i == j and i == k and j == k:
                    tenI[k,i,j] = 1
    return tenI

def ten3_parafac(A,B,C):
    # A, B and C are the factor matrices of PARAFAC decomposition ...
    I,R = A.shape
    J,R = B.shape
    K,R = C.shape
    tenX = np.zeros((K,I,J)) + 1j*np.zeros((K,I,J))
    for k in range(K):
        tenX[k,:,:] = A@np.diag(C[k,:])@B.T
    return tenX

def ten3_nmode_product(tenX,A,n):
    K,I,J = tenX.shape
    Ia,Ra = A.shape
    if n == 1:
        return ten3_fold(A@ten3_unfold(tenX,1),[Ia,J,K],1)
    elif n == 2:
        return ten3_fold(A@ten3_unfold(tenX,2),[I,Ia,K],2)
    elif n == 3:
        return ten3_fold(A@ten3_unfold(tenX,3),[I,J,Ia],3)

def ten3_tucker3(tenG,A,B,C):
    # A, B and C are the factor matrices of Tucker3 decomposition ...
    # tenG is the core tensor ...
    Kg,Ig,Jg = tenG.shape
    I,Ig = A.shape
    J,Jg = B.shape
    K,Kg = C.shape
    tenX = np.zeros((K,I,J)) + 1j*np.zeros((K,I,J))
    tenX = ten3_nmode_product(tenG,A,1)
    tenX = ten3_nmode_product(tenX,B,2)
    tenX = ten3_nmode_product(tenX,C,3)
    return tenX

def ten3_hosvd(tenX,R1,R2,R3):
    K,I,J = tenX.shape
    # Mode-1 ...
    U1,S1,Vh1 = np.linalg.svd(ten3_unfold(tenX,1), full_matrices=False)
    U1 = U1[:,0:R1]
    # Mode-2 ...
    U2,S2,Vh2 = np.linalg.svd(ten3_unfold(tenX,2), full_matrices=False)
    U2 = U2[:,0:R2]
    # Mode-3 ...
    U3,S3,Vh3 = np.linalg.svd(ten3_unfold(tenX,3), full_matrices=False)
    U3 = U3[:,0:R3]
    # Core ...
    ten_core = ten3_nmode_product(tenX,np.conjugate(U1).T,1)
    ten_core = ten3_nmode_product(ten_core,np.conjugate(U2).T,2)
    ten_core = ten3_nmode_product(ten_core,np.conjugate(U3).T,3)
    # Result (low-rank approximation) ...
    ten_hosvd = ten3_nmode_product(ten_core,U1,1)
    ten_hosvd = ten3_nmode_product(ten_hosvd,U2,2)
    ten_hosvd = ten3_nmode_product(ten_hosvd,U3,3)
    return ten_hosvd

'''
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www-users.cse.umn.edu/~saad/PDF/umsi-2006-132.pdf
'''
def ten3_hooi(tenX,R1,R2,R3,iterations):
    # Mode-2 ...
    R,_,_ = np.linalg.svd(ten3_unfold(tenX,2), full_matrices=False)
    R = R[:,0:R2]
    # Mode-3 ...
    V,_,_ = np.linalg.svd(ten3_unfold(tenX,3), full_matrices=False)
    V = V[:,0:R3]
    for i in range(int(iterations)):
        tenC = ten3_nmode_product(tenX,np.conjugate(R).T,2)
        tenC = ten3_nmode_product(tenC,np.conjugate(V).T,3)
        L,_,_ = np.linalg.svd(ten3_unfold(tenC,1), full_matrices=False)
        L = L[:,0:R1]
        tenD = ten3_nmode_product(tenX,np.conjugate(L).T,1)
        tenD = ten3_nmode_product(tenD,np.conjugate(V).T,3)
        R,_,_ = np.linalg.svd(ten3_unfold(tenD,2), full_matrices=False)
        R = R[:,0:R2]
        tenE = ten3_nmode_product(tenX,np.conjugate(L).T,1)
        tenE = ten3_nmode_product(tenE,np.conjugate(R).T,2)
        V,_,_ = np.linalg.svd(ten3_unfold(tenE,3), full_matrices=False)
        V = V[:,0:R3]
    tenB = ten3_nmode_product(tenE,np.conjugate(V).T,3)
    # Result (low-rank approximation) ...
    ten_hooi = ten3_nmode_product(tenB,L,1)
    ten_hooi = ten3_nmode_product(ten_hooi,R,2)
    ten_hooi = ten3_nmode_product(ten_hooi,V,3)
    return ten_hooi

def mls_kraof(X,ra,rb,rc):
    # X = khatri(A,B,C)
    (rx,cc) = X.shape
    A_hat = np.zeros((ra,cc)) + 1j*np.zeros((ra,cc))
    B_hat = np.zeros((rb,cc)) + 1j*np.zeros((rb,cc))
    C_hat = np.zeros((rc,cc)) + 1j*np.zeros((rc,cc))
    for i in range(cc):
        ci = X[:,i].reshape(ra,rb,rc)
        ci = np.moveaxis(ci,1,2)
        Uc,Sc,_ = np.linalg.svd(ten3_unfold(ci,1),full_matrices=False)
        Ub,Sb,_ = np.linalg.svd(ten3_unfold(ci,2),full_matrices=False)
        Ua,Sa,_ = np.linalg.svd(ten3_unfold(ci,3),full_matrices=False)
        C_hat[:,i] = (Sc[0])**(1/3) * Uc[:,0]
        B_hat[:,i] = (Sb[0])**(1/3) * Ub[:,0]
        A_hat[:,i] = (Sa[0])**(1/3) * Ua[:,0]
    return C_hat, B_hat, A_hat
    
