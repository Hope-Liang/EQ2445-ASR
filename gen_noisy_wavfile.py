# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:42:13 2022

@author: Sushanth
"""

import scipy.io.wavfile as wav
import numpy as np
from random_noise_adder import random_noise_adder
from random_noise_adder import random_noise_adder2

SNR_list = [5, 10, 15, 20, 25, 30] # the SNR for scaling the additive noise, only effective if noisy = True
noise_type_list = ["white", "pink", "babble", "hfchannel"] # additive noise type, e.g. "white", "babble"

# A sample test file
test_file = '../TIMIT/TEST/DR6/MESD0/SX372.WAV'
file_name = 'SX372'
(sample_rate, samples) = wav.read(test_file)

for SNR in SNR_list:
    for noise_type in noise_type_list:
        noisy_samples = random_noise_adder(samples, sample_rate, SNR, noise_type)
        wav.write("noisy_wav/" + file_name + '_' + str(SNR) + 'dB_' + str(noise_type),sample_rate, noisy_samples.astype(np.int16))
