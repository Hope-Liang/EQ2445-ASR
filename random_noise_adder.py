    # -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:10:55 2022

@author: Sushanth
"""

# Import required
import scipy.io.wavfile as wav
import random
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from normalize import normalize

def power_sig(signal):
    
    N = len(signal)
    p_sig = np.sum(signal**2)/N
    
    return p_sig

def random_noise_adder_and_plot(signal_file, filename, noise_file, noise_type, noise_frame_for_every_n_frames):

  n = noise_frame_for_every_n_frames

  [sample_rate, noise_samples] = wav.read(noise_file)
  [sample_rate, sig_samples] = wav.read(signal_file)

  #Normalize
  noise_samples = normalize(noise_samples)
  sig_samples = normalize(sig_samples)

  frame_ms = 25
  overlap_ms = 10
  frame_len = int(sample_rate*(frame_ms/1000))
  overlap_len = int(sample_rate*(overlap_ms/1000))

  num_sig_frames = int(len(sig_samples)/frame_len)
  num_noise_frames = int(len(noise_samples)/frame_len)

  sigmix_samples = sig_samples.copy() 
  silence = np.zeros(frame_len)

  tmp = np.arange(0, num_sig_frames, n)

  #Add noise
  for i in tmp:

    sig_idx = random.randint(0, n-1)
    sig_idx = i + sig_idx

    if(sig_idx > num_sig_frames - 1):
      break

    noise_idx = random.randint(0, num_noise_frames - 1)

    sigmix_samples[sig_idx*frame_len:sig_idx*frame_len + frame_len] = sigmix_samples[sig_idx*frame_len:sig_idx*frame_len + frame_len] + noise_samples[noise_idx*frame_len:noise_idx*frame_len + frame_len]
    #sigmix_samples[sig_idx*frame_len:sig_idx*frame_len + frame_len] = silence

  #Waveform
  t = np.arange(len(sig_samples))
  t_flt = [float(x*(1/sample_rate)) for x in t]
  plt.figure(figsize=(20, 7))
  plt.subplot(1, 2, 1)
  plt.plot(t_flt, sig_samples)
  plt.xlabel('Time [sec]')
  plt.ylabel('Amplitude')
  plt.title('Normalized Waveform - without noise')
  plt.grid()

  plt.subplot(1, 2, 2)
  plt.plot(t_flt, sigmix_samples)
  plt.xlabel('Time [sec]')
  plt.ylabel('Amplitude')
  plt.title('Normalized Waveform - with ' + filename + '_plus_' + noise_type + ' noise')
  plt.grid()
  plt.savefig('figures/' + 'waveform_' + filename + '_plus_' + noise_type + '.png')
  plt.show()

  #plot log spectrogram
  f, t, Sxx = signal.spectrogram(sig_samples, sample_rate, scaling='spectrum', window='hamm', nperseg=frame_len, noverlap=overlap_len)
  plt.figure(figsize=(20, 7))
  plt.subplot(1, 2, 1)
  plt.pcolormesh(t, f, np.log10(Sxx), shading='gourard')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.ylim(0, 8000)
  plt.grid()
  plt.title('Log Spectrogram - without noise')
  #plt.show()

  #plot log spectrogram
  f, t, Sxx = signal.spectrogram(sigmix_samples, sample_rate, scaling='spectrum', window='hamm', nperseg=frame_len, noverlap=overlap_len)
  plt.subplot(1, 2, 2)
  #plt.figure(figsize=(10, 7))
  plt.pcolormesh(t, f, np.log10(Sxx), shading='gourard')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.ylim(0, 8000)
  plt.grid()
  plt.title('Log Spectrogram - with ' + noise_type + ' noise')
  plt.savefig('figures/' + 'spectrogram_' + filename + '_plus_' + noise_type + '.png')
  plt.show()


def random_noise_adder2(samples, sample_rate, snr, noise_type):

  #default - 'white' noise
  noise_file = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/white_16kHz.wav'
  
  if noise_type == 'white':
      noise_file = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/white_16kHz.wav'
  elif noise_type == 'babble':
      noise_file = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/babble_16kHz.wav'
  elif noise_type == 'pink':
      noise_file == '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/pink_16kHz.wav'
      
  [sample_rate, noise_samples] = wav.read(noise_file)
  
  p_sig = power_sig(samples)
  p_noise = power_sig(noise_samples)

  # amplify/attenuate noise based on SNR
  # SNR = 10*log(p_sig/p_noise)
  
  gain = 10**(-snr/10)
  gain = gain*(p_sig/p_noise)
  gain = np.sqrt(gain)
  noise_samples = noise_samples*gain
  

  frame_ms = 25
  overlap_ms = 10
  frame_len = int(sample_rate*(frame_ms/1000))
  overlap_len = int(sample_rate*(overlap_ms/1000))

  num_sig_frames = int(len(samples)/frame_len)
  num_noise_frames = int(len(noise_samples)/frame_len)

  mixsig_samples = samples.copy() 
  silence = np.zeros(frame_len)

  n = 10 #add noise frame every 10 frames - to be changed
  tmp = np.arange(0, num_sig_frames, n)

  #Add noise
  for i in tmp:

    sig_idx = random.randint(0, n-1)
    sig_idx = i + sig_idx

    if(sig_idx > num_sig_frames - 1):
      break

    noise_idx = random.randint(0, num_noise_frames - 1)

    mixsig_samples[sig_idx*frame_len:sig_idx*frame_len + frame_len] = mixsig_samples[sig_idx*frame_len:sig_idx*frame_len + frame_len] + noise_samples[noise_idx*frame_len:noise_idx*frame_len + frame_len]

  return mixsig_samples

def random_noise_adder(samples, sample_rate, snr, noise_type):
    
  #default - 'white' noise
  noise_file = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/white_16kHz.wav'
  
  if noise_type == 'white':
      noise_file = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/white_16kHz.wav'
  elif noise_type == 'babble':
      noise_file = '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/babble_16kHz.wav'
  elif noise_type == 'pink':
      noise_file == '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/pink_16kHz.wav'
  elif noise_type == 'hfchannel':
      noise_type == '../NOISEX92_RawDataset/NoiseDB/NoiseX_16kHz/hfchannel_16kHz.wav'
      
  [sample_rate, noise_samples] = wav.read(noise_file)

  len_sig = len(samples)
  len_noise = len(noise_samples)
  start_idx = random.randint(0, len_noise - len_sig - 1)
  noise_samples = noise_samples[start_idx:start_idx+len_sig]
  noise_samples = np.array(noise_samples)
  
  #mean normalize signal
  mean_sig = np.mean(samples)
  std_sig = np.std(samples)
  samples = (samples - mean_sig)
  
  if noise_type == 'none':
      return samples
  
  #mean normalize and unit variance noise signal
  mean = np.mean(noise_samples)
  std = np.std(noise_samples)
  noise_samples = (noise_samples - mean)/std
  
  #mean = np.mean(noise_samples)
  #variance = np.var(noise_samples)
  #print('noise mean = ' + str(mean))
  #print('noise var = ' + str(variance))
  
  #power
  p_sig = power_sig(samples)
  p_noise = power_sig(noise_samples)

  # amplify/attenuate noise based on SNR
  # SNR = 10*log(p_sig/p_noise)
  
  gain = 10**(-snr/10)
  gain = gain*((std_sig*std_sig)/p_noise)
  gain = np.sqrt(gain)
  noise_samples = noise_samples*gain
  
  mixsig_samples = samples + noise_samples

  return mixsig_samples