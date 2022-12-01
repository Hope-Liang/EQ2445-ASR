import math
import numpy as np
import random
from scipy.io import wavfile

def read_wav(fname):
    raw_data = wavfile.read(fname)
    return raw_data[1].astype(np.float32), raw_data[0]
    
def calPower(signal):
  '''
  input: a 1D numpy array signal sequence
  output: the signal's power
  '''
  
  return sum(signal**2)/len(signal)
        
def scale(noise_type, SNR, p_signal,len_signal):
  '''
  input: noise a 1D numpy array, target SNR, original signal power, original signal length
  output: a scaled length-cut 1D numpy array noise sequence 
  '''
  noisefile=noise_type+"_16kHz.wav"
  noisefile = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/white_16kHz.wav'
  noise, fs=read_wav(noisefile)
  len_noise=len(noise)
  p_noise_temp = p_signal/(math.pow(10,SNR/10))
  mylist=np.arange(0,(len_noise-len_signal+2))
  start = random.choice(mylist)
  noise_shorten = noise[start:(start+len_signal)]
  p_noise_shorten=calPower(noise_shorten)
  return noise_shorten*((p_noise_temp/p_noise_shorten)**0.5)


def add_noise_total(signal, noise_type, SNR):
  '''
  input: 1D numpy array signal sequence, 1D numpy array noise sequence, target SNR value
  output: 1D numpy array of noisy sequence
  '''
  noisefile=noise_type+"_16kHz.wav"
  noisefile = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/white_16kHz.wav'
  noise, fs=read_wav(noisefile)
  p_signal = calPower(signal)
  len_signal = len(signal)
  noise = scale(noise_type, SNR,p_signal,len_signal)  
  #print(calPower(noise),"2")
  #print(p_signal,"3")
  return (noise+signal)
'''
def add_noise_front(signal, noise_type, SNR):
  noisefile=noise_type+"_16kHz.wav"
  noise, fs=read_wav(noisefile)
  len_signal = len(signal)
  signal_first_half = signal[:len_signal//2]
  signal_second_half = signal[len_signal//2:]
  noisy_first_half = add_noise_total(signal_first_half, noise_type, SNR)
  return np.concatenate([noisy_first_half, signal_second_half])


def add_noise_end(signal, noise_type, SNR):
  noisefile=noise_type+"_16kHz.wav"
  noise, fs=read_wav(noisefile)
  len_signal = len(signal)
  signal_first_half = signal[:len_signal//2]
  signal_second_half = signal[len_signal//2:]
  noisy_second_half = add_noise_total(signal_second_half, noise_type, SNR)
  return np.concatenate([signal_first_half, noisy_second_half])
def noise_added(signal, noise_type, SNR):
    noisefile=noise_type+"_16kHz.wav"
    noise, fs=read_wav(noisefile)
    p_signal = calPower(signal)
    len_signal = len(signal)
    noise = scale(noise_type, SNR,p_signal,len_signal)
    noise=noise[:len_signal]
    return signal+noise
'''