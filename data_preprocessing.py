import os
import tqdm
import python_speech_features
import scipy.io.wavfile as wav
import numpy as np
import random
from sklearn.model_selection import train_test_split
from six.moves import cPickle
from scipy.fftpack import dct
from random_noise_adder import random_noise_adder

##################### Feature Selection and Directory Settings ######################
# try to generate the features locally as it shall be fast, and TIMIT is a licensed dataset so we better not put
# the entire dataset folder on Google Drive. If you cannot run it locally you can modify the Dirs to do it on GoogleDrive as well
trainDir = '../TIMIT/TRAIN/' # path to training set
testDir = '../TIMIT/TEST/'
feature = 'MFCC39' # 'FilterBank123' or 'MFCC39' 
nPhonemes = 39 # 39 or 61, by default 61 as it could also be processed to 39 in the data_training pipeline
add_noise = False # False for producing clean data and True otherwise
SNR = 30 # the SNR for scaling the additive noise, only effective if add_noise = True
noise_type = "White" # additive noise type, e.g. "white", "babble"
byGroup = True # an identifier for whether to turn the preprocess data into nPhonemes groups, and 

#################### Dictionaries #########################
phoneme_map_61_39 = {
    'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 
    'nx': 'n', 'eng': 'ng', 'zh': 'sh', "ux": "uw", "pcl": "sil", "tcl": "sil", "kcl": "sil", "qcl": "sil", 
    "bcl": "sil", "dcl": "sil", "gcl": "sil", "h#": "sil", "#h": "sil", "pau": "sil", "epi": "sil", "q": "sil"
}
# a list of the 39 phonemes, check https://www.researchgate.net/publication/275055833_TCD-TIMIT_An_audio-visual_corpus_of_continuous_speech
phoneme_set_39_list = [
    'iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey', 'ay', 'oy', 'aw', 'ow', 'l', 'r', 'y', 'w', 'er', 'm', 
    'n', 'ng', 'ch', 'jh', 'dh', 'b', 'd', 'dx', 'g', 'p', 't', 'k', 'z', 'v', 'f', 'th', 's', 'sh', 'hh', 'sil'
]
# a list of the 61 phonemes, check http://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database, page 5
phoneme_set_61_list = [
    'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr','ax-h', 
    'jh', 'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng',  'em', 'nx', 
    'en', 'eng', 'l', 'r', 'w', 'y', 'hh', 'hv', 'el', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'q', 'pau', 'epi','h#'
]
# a dictionary to turn phonemes from the 61 set into number 0 to 60
phn61_to_val_dict = dict(zip(phoneme_set_61_list, list(range(61))))
# a dictionary to turn phonemes from the 39 set into number 0 to 38
phn39_to_val_dict = dict(zip(phoneme_set_39_list, list(range(39))))
# a dictionary to turn number 0 to 60 into a phoneme in the 61 set
val_to_phn61_dict = dict((v, k) for k, v in phn61_to_val_dict.items()) 
# a dictionary to turn number 0 to 38 into a phoneme in the 39 set
val_to_phn39_dict = dict((v, k) for k, v in phn39_to_val_dict.items())


##################### Functions Used #######################
def loadWavs(rootDir):
    '''
    input: rootDir of the data folder
    output: a sorted list of WAV files in the rootDir and its sub-directories
    '''
    wav_files = []
    for dirpath, dirs, files in os.walk(rootDir):
        for f in files:
            if (f.lower().endswith(".wav")):
                wav_files.append(os.path.join(dirpath, f))
    return sorted(wav_files)

def loadPhns(rootDir):
    '''
    input: rootDir of the data folder
    output: a sorted list of PHN files in the rootDir and its sub-directories
    '''
    phn_files = []
    for dirpath, dirs, files in os.walk(rootDir):
        for f in files:
            if (f.lower().endswith(".phn")):
                phn_files.append(os.path.join(dirpath, f))
    return sorted(phn_files)

def loadData(rootDir):
    '''
    input: rootDir of the data folder
    output: a tuple of two sorted lists for WAV, PHN files in the rootDir and its sub-directories
    '''
    wav_files = loadWavs(rootDir)
    label_files = loadPhns(rootDir)
    print("==== In {}====".format(rootDir))
    print("Found %d WAV files" % len(wav_files))
    print("Found %d PHN files" % len(label_files))
    return wav_files, label_files

def calFeature(signal, rate, winlen=0.025, winstep=0.01, nfilt=40, preemph=0.97, appendEnergy=True):
    # Code from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    '''
    input: signal, sample_rate
    output: FBank (40-dim) and MFCC (12-dim) features 
    '''
    emphasized_signal = np.append(signal[0], signal[1:] - preemph * signal[:-1])
    signal_length = len(emphasized_signal)
    frame_length, frame_step = int(round(winlen * rate)), int(round(winstep * rate))  # Convert from seconds to samples
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    frames *= np.hamming(frame_length)
    
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    binn = np.floor((NFFT + 1) * hz_points / rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(binn[m - 1])   # left
        f_m = int(binn[m])             # center
        f_m_plus = int(binn[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - binn[m - 1]) / (binn[m] - binn[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (binn[m + 1] - k) / (binn[m + 1] - binn[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    cep_lifter = 22
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  #*
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    if appendEnergy:
        filter_banks = filter_banks # needs to be further implemented but this function is not used at the moment
    
    return filter_banks, mfcc

def extractFeatures(filename, feature, add_noise=False, SNR=None, noise_type=None):
    '''
    input: 
        filename: a WAV filename
        feature: feature to be extracted (available from "MFCC39" and "FilterBank123")
        add_noise: whether to add noise to the clean file before feature extration
        SNR: scaling factor used for additive noise
        noise_type: a string containing the type of additive noise
    output: 
        a numpy array (2D) containing the extracted features for the input WAV file
    '''
    (rate, sample) = wav.read(filename)
    #subtract the signal by its mean
    mean_sig = np.mean(sample)
    sample = (sample - mean_sig)
    
    # specify snr and enable code below to add noise
    if add_noise:
        sample = random_noise_adder(sample, rate, SNR, noise_type)
    
    if feature == 'MFCC39':
        # extracts 39-dim MFCC features
        mfcc = python_speech_features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, preemph=0.97, appendEnergy=True)
        derivative1 = np.zeros(mfcc.shape)
        for i in range(1, mfcc.shape[0] - 1):
            derivative1[i, :] = mfcc[i+1, :] - mfcc[i-1, :]
        derivative2 = np.zeros(derivative1.shape)
        for i in range(1, derivative1.shape[0] - 1):
            derivative2[i, :] = derivative1[i+1, :] - derivative1[i-1, :]
        features = np.concatenate((mfcc, derivative1, derivative2), axis=1) # (13+13+13)-dim
        
    elif feature == 'FilterBank123':
        # extracts 123-dim filter-bank features
        fbank,energy = python_speech_features.fbank(sample, rate, winlen=0.025, winstep=0.01, nfilt=40, nfft=512, lowfreq=0, highfreq=None, preemph=0.97)
        fb = np.c_[energy, fbank]
        derivative1 = np.zeros(fb.shape)
        for i in range(1, fb.shape[0] - 1):
            derivative1[i, :] = fb[i+1, :] - fb[i-1, :]
        derivative2 = np.zeros(derivative1.shape)
        for i in range(1, derivative1.shape[0] - 1):
            derivative2[i, :] = derivative1[i+1, :] - derivative1[i-1, :]
        features = np.concatenate((fb, derivative1, derivative2), axis=1) # (41+41+41)-dim       
    else:
        print('ERROR: Feature {} not available!'.format(feature))
        features = None      
    return features

def get_total_duration(filename):
    """Get the length of the phoneme file, i.e. the 'time stamp' of the last phoneme"""
    for line in reversed(list(open(filename))):
        [_, val, _] = line.split()
        return int(val)

def preprocessData(rootDir, feature, nPhonemes, add_noise=False, SNR=None, noise_type=None):
    '''
    input: 
        rootDir: root directory of the data folder, 
        feature: feature to be extracted (available from "MFCC39" and "FilterBank123")
        nPhonemes: #Phonemes to be predicted (39 or 61)
        add_noise: whether to add noise to the clean data, by default false
        SNR: the SNR for added noise factors
    output: 
        X - a list of 2-D numpy array, containing the extracted features of different frame length
        y - a list of 1-D numpy array, containing the corresponding labels
    '''
    wav_files, label_files = loadData(rootDir)
    X = []
    y = []
    for i in range(len(wav_files)):
        phn_name = str(label_files[i])
        wav_name = str(wav_files[i])
        if (wav_name.startswith("SA")):  # specific for TIMIT: these files contain strong dialects; don't use them
            continue # this actually did nothing as wav_name is the absolute path in the system, e.g. of the format'/Users/lxy/Desktop/TIMITspeech/TIMIT/TEST/FADG0/SA1.WAV'
        X_wav = extractFeatures(wav_name, feature, add_noise, SNR, noise_type)
        
        total_duration = get_total_duration(phn_name)
        total_frames = X_wav.shape[0]
        
        y_wav = np.zeros(total_frames) - 1 # initialized as -1
        
        fr = open(phn_name)
        first_record = True # this is used to handle the cases that in the PHN file, some records at the beginning have missing labels, e.g. 'TIMIT/TRAIN/DR1/FSAH0/SI614.PHN'
        for line in fr:
            [start_time, end_time, phoneme] = line.rstrip('\n').split()
            start_time = int(start_time)
            end_time = int(end_time)
            if first_record == True and start_time != 0:
                start_time = 0
            first_record = False
            
            start_ind = int(np.round(start_time / (total_duration / total_frames)))
            end_ind = int(np.round(end_time / (total_duration / total_frames)))       
            
            if nPhonemes == 39:    
                if (phoneme in phoneme_map_61_39.keys()):
                    phoneme = phoneme_map_61_39[phoneme]
                phoneme_num = phn39_to_val_dict[phoneme]
                y_wav[start_ind:end_ind] = phoneme_num
            elif nPhonemes == 61:
                phoneme_num = phn61_to_val_dict[phoneme]
                y_wav[start_ind:end_ind] = phoneme_num
            else:
                print('ERROR: #Phonemes {} not supported!'.format(nPhonemes))
                return None, None
        fr.close()
            
        if sum(y_wav==-1) != 0:
            print(y_wav)
        assert sum(y_wav==-1) == 0

        X.append(X_wav)
        y.append(y_wav)
    return X, y

def calc_norm_param(X):
    """Assumes X to be a list of arrays (of differing sizes)"""
    total_len = 0
    mean_val = np.zeros(X[0].shape[1])
    std_val = np.zeros(X[0].shape[1])
    for obs in X:
        obs_len = obs.shape[0]
        mean_val += np.mean(obs, axis=0) * obs_len
        std_val += np.std(obs, axis=0) * obs_len
        total_len += obs_len

    mean_val /= total_len
    std_val /= total_len

    return mean_val, std_val, total_len

def normalizeData(X):
    """Assumes X to be a list of arrays (of differing sizes)"""
    mean_val, std_val, _ = calc_norm_param(X)
    for i in range(len(X)):
        X[i] = (X[i] - mean_val) / std_val
    return X

def sparseData(X, y, nPhonemes):
    '''
    input:
        X: a list of 2D numpy arrays (features)
        y: a list of 1D numy arrays (labels)
        nPhonemes: the number of phonemes used in the labels
    output:
        X_ret: a dictionary with keys from 0 to nPhonemes-1 (the labels) and values for the feature segments
        y_ret: a dictionary with keys from 0 to nPhonemes-1 (the labels) and values for the label segments
    '''
    X_ret = {}
    y_ret = {}
    ttl_phn_ori = 0
    ttl_phn_grp = 0

    for val in range(nPhonemes):
        X_ret[val] = []
        y_ret[val] = []

    for i in range(len(y)):
        ttl_phn_ori += y[i].shape[0]
        left = 0
        for right in range(1,len(y[i])):
            if y[i][right-1] != y[i][right]:
                X_ret[int(y[i][left])].append(X[i][left:right])
                y_ret[int(y[i][left])].append(y[i][left:right])
                ttl_phn_grp += (right-left)
                left = right
            if right==len(y[i])-1:
                X_ret[int(y[i][left])].append(X[i][left:])
                y_ret[int(y[i][left])].append(y[i][left:])
                ttl_phn_grp += (right-left+1)

    assert ttl_phn_ori == ttl_phn_grp

    return X_ret, y_ret

def train_test_split_byGroup(X_train, y_train, nPhonemes=39):
    X_train_ret = {}
    y_train_ret = {}
    X_val_ret = {}
    y_val_ret = {}
    for i in range(nPhonemes):
        X_train_temp, X_val_temp, y_train_temp, y_val_temp = train_test_split(X_train[i], y_train[i], test_size=0.1, random_state = 42)
        X_train_ret[i] = X_train_temp
        y_train_ret[i] = y_train_temp
        X_val_ret[i] = X_val_temp
        y_val_ret[i] = y_val_temp

    return X_train_ret, X_val_ret, y_train_ret, y_val_ret


def saveDataToPkl(target_path, data):  # data can be list or dictionary, save data at target_path (shall end in .pkl)
    if not os.path.exists(os.path.dirname(target_path)):
        os.makedirs(os.path.dirname(target_path))
    with open(target_path, 'wb') as cPickle_file:
        cPickle.dump(data, cPickle_file, protocol=5) # protocol = 2 for py2
    return 0


######################## Actual Processing #####################
if not byGroup:
    X_train, y_train = preprocessData(trainDir, feature, nPhonemes, add_noise = add_noise, SNR = SNR, noise_type=noise_type)
    X_test, y_test = preprocessData(testDir, feature, nPhonemes, add_noise = add_noise, SNR = SNR, noise_type=noise_type)
    X_train = normalizeData(X_train)
    X_test = normalizeData(X_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state = 42) # a random seed is fixed so the data order will be kept exactly the same no matter how many runs
    # if add_noise = True, save only the test data, otherwise save all data
    if add_noise:
        dataList = [X_test, y_test]
        savename = "../TIMIT_"+feature+"_nPhonemes"+str(nPhonemes)+"_noisy"+str(SNR)+str(noise_type)+".pkl"
    else:
        dataList = [X_train, y_train, X_val, y_val, X_test, y_test]
        savename = "../TIMIT_"+feature+"_nPhonemes"+str(nPhonemes)+"_clean.pkl"
    saveDataToPkl(savename, dataList)
else:
    X_train, y_train = preprocessData(trainDir, feature="MFCC39", nPhonemes=39)
    X_test, y_test = preprocessData(testDir, feature="MFCC39", nPhonemes=39)
    X_train, y_train = sparseData(X_train, y_train, nPhonemes=39)
    X_test, y_test = sparseData(X_test, y_test, nPhonemes=39)
    X_train, X_val, y_train, y_val = train_test_split_byGroup(X_train, y_train)
    #for i in range(39):
        #print("train/val for phoneme {}: {}".format(i, len(X_train[i])/len(X_val[i])))
    dataList = [X_train, y_train, X_val, y_val, X_test, y_test]
    savename = "../TIMIT_"+feature+"_nPhonemes39_clean_byGroup.pkl"
    saveDataToPkl(savename, dataList)
    