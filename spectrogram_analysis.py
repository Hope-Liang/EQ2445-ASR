# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:50:46 2022

@author: Sushanth
"""

# Imports needed

import os
import scipy.io.wavfile as wav
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import random
from random_noise_adder import random_noise_adder_and_plot

def normalize(samples):
  samples_norm = samples/max(samples)
  return samples_norm

def plot_td_fd(file, filename):
  (sample_rate, samples) = wav.read(file)
  print('Sample rate =' + str(sample_rate))

  frame_ms = 25
  overlap_ms = 10
  frame_len = int(sample_rate*(frame_ms/1000))
  overlap_len = int(sample_rate*(overlap_ms/1000))

  #time-domain
  t = np.arange(len(samples))
  t_flt = [float(x*(1/sample_rate)) for x in t]
  #t_flt = t_flt*(1/sample_rate)
  plt.figure(figsize=(10, 7))
  #samples = normalize(samples)
  plt.plot(t_flt, samples)
  plt.xlabel('Time [sec]')
  plt.ylabel('Amplitude')
  plt.title('Normalized Waveform - ' + filename)
  plt.grid()
  plt.savefig('figures/' + 'waveform_' + filename + '.png')
  
  #freq-domain
  f, t, Sxx = signal.spectrogram(samples, sample_rate, scaling='spectrum', window='hamm', nperseg=frame_len, noverlap=overlap_len)
  #print(max(Sxx.any()))
  plt.figure(figsize=(10, 7))
  plt.pcolormesh(t, f, np.log10(Sxx), shading='gourard')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.ylim(0, 8000)
  plt.grid()
  plt.title('Log Spectrogram - ' + filename)
  plt.savefig('figures/' + 'spectrogram_' + filename + '.png')
  plt.show()

# A sample train file
train_file = '../TIMIT/TRAIN/DR3/FALK0/SX366.WAV'
plot_td_fd(train_file, 'SX366')


# A sample test file
test_file = '../TIMIT/TEST/DR6/MESD0/SX372.WAV'
plot_td_fd(test_file, 'SX372')

# White Noise
test_file = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/white_16kHz.wav'
plot_td_fd(test_file, 'white_16kHz')

# Pink Noise
test_file = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/pink_16kHz.wav'
plot_td_fd(test_file, 'pink_16kHz')

# Babble Noise
test_file = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/babble_16kHz.wav'
plot_td_fd(test_file, 'babble_16kHz')

# white noise + signal
noise_file = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/white_16kHz.wav'
signal_file = '../TIMIT/TRAIN/DR3/FALK0/SX366.WAV'
random_noise_adder_and_plot(signal_file,  'SX366', noise_file, 'white_noise', 20)

# pink noise + signal
noise_file = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/pink_16kHz.wav'
signal_file = '../TIMIT/TRAIN/DR3/FALK0/SX366.WAV'
random_noise_adder_and_plot(signal_file, 'SX366', noise_file, 'pink_noise', 10)
