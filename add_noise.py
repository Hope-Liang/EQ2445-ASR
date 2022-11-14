import math
import numpy as np


def calPower(signal):
  '''
  input: a 1D numpy array signal sequence
  output: the signal's power
  '''
  return sum(signal**2)/len(signal)


def scale(noise, SNR, p_signal, len_signal):
  '''
  input: noise a 1D numpy array, target SNR, original signal power, original signal length
  output: a scaled length-cut 1D numpy array noise sequence 
  '''
  p_noise_temp = p_signal/(math.exp(snr/10))
  noise_shorten = noise[:len_signal]
  p_noise = calPower(noise_shorten)
  return noise_shorten*((p_noise_temp/p_noise)**0.5)


def add_noise(signal, noise, SNR):
  '''
  input: 1D numpy array signal sequence, 1D numpy array noise sequence, target SNR value
  output: 1D numpy array of noisy sequence
  '''
  p_signal = calPower(signal)
  len_signal = len(signal)
  noise = scale(noise, SNR, p_signal, len_signal)
  noise_temp = np.zeros(len_signal)
  nums=sorted(np.random.choice(l, 20))
  print(nums)
  for i in range(0,20,2):
    for j in range(nums[i], nums[i+1]):
      noise_1[j]=noise[j]
  return (noise_1+signal)


def add_noise_front(signal, noise, SNR):
  len_signal = len(signal)
  signal_first_half = signal[:len_signal//2]
  signal_second_half = signal[len_signal//2:]
  noisy_first_half = add_noise(signal_first_half, noise, SNR)
  return np.concatenate([noisy_first_half, signal_second_half])


def add_noise_end(signal, noise, SNR):
  len_signal = len(signal)
  signal_first_half = signal[:len_signal//2]
  signal_second_half = signal[len_signal//2:]
  noisy_second_half = add_noise(signal_second_half, noise, SNR)
  return np.concatenate([signal_first_half, noisy_second_half])
