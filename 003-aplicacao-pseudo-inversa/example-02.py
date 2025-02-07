'''
MIMO system example:
    M - tx antennas)
    N - rx antennas)
    T - symbol periods)
Problem: pilot assisted channel and symbol estimation
'''

import numpy as np
import matplotlib.pyplot as plt

def awgn_noise(signal_matrix,snr_db):
    noise = (1/np.sqrt(2)) * (np.random.normal(0,1,signal_matrix.shape) +1j*np.random.normal(0,1,signal_matrix.shape))
    alpha = np.sqrt(((np.linalg.norm(signal_matrix,'fro')**2) / (np.linalg.norm(noise,'fro')**2)) * np.power(10,-snr_db/10))
    return signal_matrix + alpha*noise

def nmse(Ao, Ahat):
    return (np.linalg.norm(Ahat - Ao, 'fro')**2) / (np.linalg.norm(Ao, 'fro')**2)

# Parameters ...
M = 2
N = 4
T = 50
SNR = np.array(range(-5,35,5))
monte_carlo = int(1e+3)

NMSE_H = []
BER = []
for snr in SNR:
    print(f"SNR = {snr}dB")
    ber = []
    nmse_H = []
    for runs in range(monte_carlo):
        Ho = (1/np.sqrt(2)) * (np.random.normal(0,1,(N,M)) + 1j*np.random.normal(0,1,(N,M))) # channel matrix
        So = np.random.choice([-1,1],(M,T)) # pilots matrix
        Y = awgn_noise(np.dot(Ho,So),snr)

        # Pilot-assisted channel estimation ...
        Hhat = np.dot(Y,np.linalg.pinv(So))
        nmse_H.append(nmse(Ho,Hhat))

        # Data estimation ...
        D = np.random.choice([-1,1],(M,T)) # data matrix
        Y = awgn_noise(np.dot(Ho,D),snr)
        Dhat = np.sign(np.real(np.dot(np.linalg.pinv(Hhat),Y)))

        # BER computation ...
        ber.append(np.count_nonzero(D - Dhat)/(M*T))

    NMSE_H.append(np.mean(nmse_H))
    BER.append(np.mean(ber))

# Plots ...
fig, nmse = plt.subplots()
plt.yscale('log')
nmse.plot(SNR,NMSE_H,marker='o')
nmse.set_xlabel("SNR[dB]")
nmse.set_ylabel("NMSE(H)")
nmse.set_title("NMSE x SNR")
plt.xlim(min(SNR), max(SNR))
plt.grid()
plt.show()

fig, ber = plt.subplots()
plt.yscale('log')
ber.plot(SNR,BER,marker='o')
ber.set_xlabel("SNR[dB]")
ber.set_ylabel("BER")
ber.set_title("BER x SNR")
plt.xlim(min(SNR), max(SNR))
plt.grid()
plt.show()









