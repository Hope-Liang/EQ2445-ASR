def p(signal):
  sum=0
  for k in range(len(signal)):
    sum+=signal[k]*signal[k]
  return (sum/len(signal))

def scale(signal, noise, snr):
  p_signal=p(signal)
  p_noise_1=p_signal/(math.exp(snr/10))
 
  noise_shorten=noise[0:len(signal)]
  p_noise_0=p(noise_shorten)
  noise_1=noise_shorten*((p_noise_1/p_noise_0)**0.5)
  return noise_1

def add_noise(signal, noise, snr):
  noise=scale(signal, noise, snr)
  l=len(noise)
  noise_1=[0 for _ in range(l)]
  nums=sorted(np.random.choice(l, 20))
  print(nums)
  for i in range(0,20,2):
    for j in range(nums[i], nums[i+1]):
      noise_1[j]=noise[j]
  return (noise_1+signal)

def add_noise_front(signal, noise, snr):
  noise=scale(signal, noise, snr)
  l=len(noise)//2
  noise_1=[0 for _ in range(len(noise))]
  nums=sorted(np.random.choice(l, 20))
  for i in range(0,20,2):
    for j in range(nums[i], nums[i+1]):
      noise_1[j]=noise[j]
  return (noise_1+signal)

def add_noise_end(signal, noise, snr):
  noise=scale(signal, noise, snr)
  l=len(noise)//2
  noise_1=[0 for _ in range(len(signal))]
  nums=sorted(np.random.choice(range(l,len(signal)), 20))
  
  for i in range(0,20,2):
    for j in range(nums[i], nums[i+1]):
      noise_1[j]=noise[j]
  return (noise_1+signal)