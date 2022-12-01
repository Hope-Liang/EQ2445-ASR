# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:07:51 2022

@author: Sushanth
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pickle as pkl
import numpy as np

with open('snr_noise_acc.txt', 'rb') as f:
    snr_noise_accu_bi_lstm = pkl.load(f)

snr_noise_accu_bi_lstm = snr_noise_accu_bi_lstm*100
#remove 5dB and 30dB
print(snr_noise_accu_bi_lstm)
#snr_noise_accu = np.delete(snr_noise_acc, 0, 1)
#snr_noise_accu_bi_lstm = np.delete(snr_noise_accu, 4, 1)*100

snr_list = [10, 15, 20, 25] # the SNR for scaling the additive noise, only effective if noisy = True
noise_type_list = ["white", "pink", "babble", "hfchannel"] # additive noise type, e.g. "white", "babble"

#GMM-HMM
snr_noise_accu_gmm_hmm = np.array([[27.9, 36.8, 46.8, 55.6],
                          [32.2,  42.3, 51.9, 59.9],
                          [37.4, 49.3, 59.3, 65.7],
                          [33.3, 44.1, 54.4, 62.3]])

#NVP-HMM
snr_noise_accu_nvp_hmm = np.array([[37.7, 49.4, 60.0, 67.1],
                          [33.7, 48.6, 61.7, 69.3],
                          [42.3, 56.2, 65.8, 70.7],
                          [44.9, 55.8, 63.4, 67.9]])





ref_acc_bi_lstm = np.ones([4, 1], dtype=float)*77.85
ref_acc_gmm_hmm = np.ones([4, 1], dtype=float)*72.8
ref_acc_nvp_hmm = np.ones([4, 1], dtype=float)*77.6

ref_acc_bi_lstm_mat = np.ones([4, 4], dtype=float)*77.85
ref_acc_gmm_hmm_mat = np.ones([4, 4], dtype=float)*72.8
ref_acc_nvp_hmm_mat = np.ones([4, 4], dtype=float)*77.6

#drop in accu
drop_accu_bi_lstm = ref_acc_bi_lstm_mat - snr_noise_accu_bi_lstm
drop_accu_nvp_hmm = ref_acc_nvp_hmm_mat - snr_noise_accu_nvp_hmm

plt.rcParams['font.size'] = '11'

#white noise
figure(figsize=(7, 5), dpi=300)
plt.plot(snr_list, ref_acc_bi_lstm, 'r', label = "Bi-LSTM ref acc", linestyle="--")
plt.plot(snr_list, ref_acc_gmm_hmm, 'g', label = "GMM-HMM ref acc", linestyle="--")
plt.plot(snr_list, ref_acc_nvp_hmm, 'b', label = "NVP-HMM ref acc", linestyle="--")

plt.plot(snr_list, snr_noise_accu_bi_lstm[0,:], 'r', marker='*', label = "Bi-LSTM snr acc", linestyle="-")
plt.plot(snr_list, snr_noise_accu_gmm_hmm[0,:], 'g', marker='*', label = "GMM-HMM snr acc", linestyle="-")
plt.plot(snr_list, snr_noise_accu_nvp_hmm[0,:], 'b', marker='*', label = "NVP-HMM snr acc", linestyle="-")

plt.xlabel('SNR(dB)')
plt.ylabel('accuracy(%)')
plt.title('White Noise')
plt.legend()
plt.grid()
plt.savefig('white_noise.png',bbox_inches='tight')
plt.show()



#pink noise
figure(figsize=(6, 4.5), dpi=300)
plt.plot(snr_list, ref_acc_bi_lstm, 'r', label = "Bi-LSTM ref acc", linestyle="--")
plt.plot(snr_list, ref_acc_gmm_hmm, 'g', label = "GMM-HMM ref acc", linestyle="--")
plt.plot(snr_list, ref_acc_nvp_hmm, 'b', label = "NVP-HMM ref acc", linestyle="--")

plt.plot(snr_list, snr_noise_accu_bi_lstm[1,:], 'r', marker='D', label = "Bi-LSTM snr acc", linestyle="-")
plt.plot(snr_list, snr_noise_accu_gmm_hmm[1,:], 'g', marker='D', label = "GMM-HMM snr acc", linestyle="-")
plt.plot(snr_list, snr_noise_accu_nvp_hmm[1,:], 'b', marker='D', label = "NVP-HMM snr acc", linestyle="-")

plt.xlabel('SNR(dB)')
plt.ylabel('accuracy(%)')
plt.title('Pink Noise')
plt.legend()
plt.grid()
plt.savefig('pink_noise.png',bbox_inches='tight')
plt.show()


#babble noise
figure(figsize=(7, 5), dpi=300)
plt.plot(snr_list, ref_acc_bi_lstm, 'r', label = "Bi-LSTM ref acc", linestyle="--")
plt.plot(snr_list, ref_acc_gmm_hmm, 'g', label = "GMM-HMM ref acc", linestyle="--")
plt.plot(snr_list, ref_acc_nvp_hmm, 'b', label = "NVP-HMM ref acc", linestyle="--")

plt.plot(snr_list, snr_noise_accu_bi_lstm[2,:], 'r', marker='D', label = "Bi-LSTM snr acc", linestyle="-")
plt.plot(snr_list, snr_noise_accu_gmm_hmm[2,:], 'g', marker='D', label = "GMM-HMM snr acc", linestyle="-")
plt.plot(snr_list, snr_noise_accu_nvp_hmm[2,:], 'b', marker='D', label = "NVP-HMM snr acc", linestyle="-")

plt.xlabel('SNR(dB)')
plt.ylabel('accuracy(%)')
plt.title('Babble Noise')
plt.legend()
plt.grid()
plt.savefig('babble_noise.png',bbox_inches='tight')
plt.show()

#hfchannel noise
figure(figsize=(6, 4.5), dpi=300)
plt.plot(snr_list, ref_acc_bi_lstm, 'r', label = "Bi-LSTM ref acc", linestyle="--")
plt.plot(snr_list, ref_acc_gmm_hmm, 'g', label = "GMM-HMM ref acc", linestyle="--")
plt.plot(snr_list, ref_acc_nvp_hmm, 'b', label = "NVP-HMM ref acc", linestyle="--")

plt.plot(snr_list, snr_noise_accu_bi_lstm[3,:], 'r', marker='D', label = "Bi-LSTM snr acc", linestyle="-")
plt.plot(snr_list, snr_noise_accu_gmm_hmm[3,:], 'g', marker='D', label = "GMM-HMM snr acc", linestyle="-")
plt.plot(snr_list, snr_noise_accu_nvp_hmm[3,:], 'b', marker='D', label = "NVP-HMM snr acc", linestyle="-")

plt.xlabel('SNR(dB)')
plt.ylabel('accuracy(%)')
plt.title('Hfchannel Noise')
plt.legend()
plt.grid()
plt.savefig('hfchannel_noise.png',bbox_inches='tight')
plt.show()


#drop in accuracy
#Bi-LSTM
figure(figsize=(6, 4.5), dpi=300)

plt.plot(snr_list, drop_accu_bi_lstm[0,:], 'r', marker='*', label = "white", linestyle="-", markersize=10)
plt.plot(snr_list, drop_accu_bi_lstm[1,:], 'g', marker='D', label = "pink", linestyle="-", markersize=10)
plt.plot(snr_list, drop_accu_bi_lstm[2,:], 'b', marker='o', label = "babble", linestyle="-", markersize=10)
plt.plot(snr_list, drop_accu_bi_lstm[3,:], 'y', marker='P', label = "hf-channel", linestyle="-", markersize=10)

plt.xlabel('SNR(dB)')
plt.ylabel('drop in accuracy(%)')
plt.title('Drop in accuracy of Bi-LSTM model')
plt.legend()
plt.grid()
plt.savefig('drop_aacu_bi_lstm.png',bbox_inches='tight')
plt.show()

#NVP-HMM
figure(figsize=(6, 4.5), dpi=300)

plt.plot(snr_list, drop_accu_nvp_hmm[0,:], 'r', marker='*', label = "white", linestyle="-", markersize=10)
plt.plot(snr_list, drop_accu_nvp_hmm[1,:], 'g', marker='D', label = "pink", linestyle="-", markersize=10)
plt.plot(snr_list, drop_accu_nvp_hmm[2,:], 'b', marker='o', label = "babble", linestyle="-", markersize=10)
plt.plot(snr_list, drop_accu_nvp_hmm[3,:], 'y', marker='P', label = "hf-channel", linestyle="-", markersize=10)

plt.xlabel('SNR(dB)')
plt.ylabel('drop in accuracy(%)')
plt.title('Drop in accuracy of NVP-HMM model')
plt.legend()
plt.grid()
plt.savefig('drop_aacu_nvp_hmm.png',bbox_inches='tight')
plt.show()