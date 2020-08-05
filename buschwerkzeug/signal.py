import scipy
import scipy.signal as dummy
import numpy as np
import pandas as pd
import noisereduce
import math

def rms(wav):
    return np.mean(np.power(wav,2))**(1/2.)

def spectrogram(wav, fs, win_len=None, overlap = None, log = True):
    win_type = 'hann'
    if not win_len:
        win_len = 2**round(math.log(0.01*fs, 2)) # power of two closest to 10 ms
        win_len = min(win_len, len(wav))
    if not overlap:
        overlap = win_len//2
    f, t, S = scipy.signal.stft(wav, fs, window = win_type, nperseg=win_len, noverlap=overlap)
    S = S[1:]
    f = f[1:]
    S=np.abs(S)
    if log:
        S=np.log(S+1e-10) 
    return f, t, S

##def scale_spectrogram(S, log_scaling_factor=10, traget_len):
#    S = Image.fromarray(S).resize([ int(np.log(S.shape[1])

# Scaling from AVGN paper:
# https://github.com/timsainb/avgn_paper
def log_resize_spec(spec, scaling_factor=10):
    resize_shape = [int(np.log(np.shape(spec)[1]) * scaling_factor), np.shape(spec)[0]]
    resize_spec = np.array(Image.fromarray(spec).resize(resize_shape, Image.ANTIALIAS))
    return resize_spec


# Pading from AVGN paper:
# https://github.com/timsainb/avgn_paper
def pad_spectrogram(spectrogram, pad_length):
    """ Pads a spectrogram to being a certain length
    """
    excess_needed = pad_length - np.shape(spectrogram)[1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    return np.pad( spectrogram, [(0, 0), (pad_left, pad_right)], "constant", constant_values=0 )

def median_dominant_freq(wav, fs, freq_min, freq_max):
    f,t,S = spectrogram(wav, fs)
    f_idx = np.logical_and(freq_min <= f, f <= freq_max)
    f = f[f_idx]
    S = S[f_idx, :]
    return f[int(np.median(np.argmax(S, axis=0)))]

def bandpass(wav, low, high, fs):
    sos = scipy.signal.butter(1, (low, high), 'bandpass', fs=fs, output='sos')
    return scipy.signal.sosfilt(sos, wav)

def envelope(wav, fs, window_len, smooth_window_len = None):
    if not smooth_window_len:
        smooth_window_len = window_len
    env = scipy.ndimage.maximum_filter(np.abs(wav), size=window_len)
    #env = scipy.ndimage.uniform_filter(env, window_len)
    #env = scipy.ndimage.gaussian_filter(env, smooth_window_len,  mode='constant')
    return env

def spectral_entropy(wav, fs, win_len=None, overlap = None, smooth_window_len = None):
    f,t,S = spectrogram(wav, fs, win_len=None, overlap = None, log = False)
    S/=S.sum(axis=0)[np.newaxis, :]
    se = -np.multiply(S, np.log2(S)).sum(axis=0)
    se /= np.log2(S.shape[0])
    se = scipy.signal.resample(se, len(wav))
    return se
