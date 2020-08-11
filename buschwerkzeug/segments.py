import scipy
import scipy.signal as dummy
import numpy as np
import pandas as pd
import noisereduce
import skimage
from skimage import filters as dummy
from . import signal
import soundfile as sf
from functools import lru_cache
from pathlib import PurePath

load_wav = lru_cache(1)(sf.read)

def dummy_segmenter(segments):
    def f(fname):
        return segments[PurePath(fname).name]
    return f

def spectral_entropy_segmenter(env_win_len, hold_len, min_len, threshold_method='otsu_exp', cutoff = 0):
    def f(fname):
        wav, fs = load_wav(fname)
        env = 1 - signal.spectral_entropy(wav, fs, env_win_len)
        return segments(env, hold_len, min_len, threshold_method, cutoff, env_win_len)
    return f

def amplitude_segmenter(env_window_len, hold_len, min_len, threshold_method='otsu_exp', cutoff = 0):
    def f(fname):
        wav, fs = load_wav(fname)
        env = signal.envelope(wav, fs, env_window_len)
        return segments(env, hold_len, min_len, threshold_method, cutoff, env_window_len)
    return f


def segments(env, hold_len, min_len, threshold_method='otsu_exp', cutoff_low = 0, descend_hold_len = 0, join_after_descend = False):
    cutoff_env = env[env >= cutoff_low]
    if len(cutoff_env) < min_len:
        segments = []
    else:
        th = threshold(cutoff_env, threshold_method)
        segments = detect(env, hold_len, th)
        if descend_hold_len:
            segments = list(map(lambda b: descend(env, *b, descend_hold_len), segments))
        if join_after_descend:
            segments = join(segments, hold_len)
        segments = filter_min_len(segments, min_len) 
    return pd.DataFrame(segments, columns=('start', 'end'))

def filter_min_len(segments, min_len):
    return list(filter(lambda s: s[1]-s[0]>=min_len, segments))

def threshold(env, method=False):
    if not method or method =='otsu':
        return skimage.filters.threshold_otsu(env)
    elif method == 'otsu_exp':
        return np.exp(skimage.filters.threshold_otsu(np.log(env+1e-20)))
    elif isinstance(method, float):
        return method
    else:
        assert(False)

def detect(wav, hold_len, th):
    wav = np.abs(wav)
    idx, = np.diff(wav>th).nonzero()
    if wav[0] > th:
        idx=np.insert(idx, 0, 0)
    if wav[-1] > th:
        idx=np.append(idx, len(wav)-1)
    idx.shape = (-1,2)
    segments = []
    if len(idx):
        start,end = idx[0]
    for i in range(1,len(idx)):
        if idx[i][0] - end > hold_len:
            segments.append((start, end))
            start = idx[i][0]
        end = idx[i][1]
    segments.append((start, end))
    return segments

def descend(env, start, end, hold_len):

    hold = 0

    while start>0 and env[start-1] <= env[start] and hold < hold_len:
        if env[start-1] < env[start]:
            hold = 0
        else:
            hold += 1
        start-=1
    if hold == hold_len:
        start += hold

    while end<len(env) and env[end-1] >= env[end] and hold < hold_len:
        if env[end-1] > env[end]:
            hold = 0
        else:
            hold += 1
        end += 1
    if hold == hold_len:
        end-=hold

    return start, end

def join(segments, hold_len):
    i = 1
    while i < len(segments):
        if segments[i][0] - segments [i-1][1] <= hold_len:
            segments[i][0] = segments[i-1][0]
            del segments[i-1]
        else:
            i+=1
    return segments




def match(segments1, segments2, tolerance_ratio):
    segments1['match'] = 0
    segments2['match'] = 0

    i = 0
    for segment in segments1.itertuples():
        tolerance = tolerance_ratio * (segment.end-segment.start)
        match = (
            (segments2.fname==segment.fname) &
            (np.abs(segments2.start-segment.start)<=tolerance) &
            (np.abs(segments2.end-segment.end)<=tolerance)
        )
        if match.sum() > 0:
            segments1.loc[i,'match'] += 1
            segments2.loc[match, 'match'] += 1
        i += 1
    return segments1.match.values, segments2.match.values

def match_score(prediction, control, tolerance_ratio):
    prediction, control = match(prediction, control, tolerance_ratio)
    assert((prediction<2).all() and (control<2).all())
    if sum(prediction) is 0:
        precision = 1
    else:
        precision = sum(prediction)/len(prediction)
    recall = sum(control)/len(control)
    num = precision + recall
    if not num:
        f1score = 0
    else:
        f1score = 2*precision*recall/(precision+recall)
    return f1score, precision, recall

def score(control, prediction, tolerance, missed_table = None):
    n_control = sum(map(len, control))
    n_prediction = sum(map(len, prediction))
    tp = 0
    missed = pd.DataFrame()
    for prediction_segments, control_segments in zip(prediction, control):
        for segment in control_segments.itertuples():
            if np.any( 
                    (np.abs(prediction_segments.start-segment.start)<=tolerance) &
                    (np.abs(prediction_segments.end-segment.end)<=tolerance)
                    ):
                tp += 1
            else:
                #missed = missed.append(segment._asdict(), ignore_index=True)
                pass
    #missed.to_csv('/tmp/missed.csv')
    #print(n_control, n_prediction, tp)
    precision = 1 if not n_prediction else tp / n_prediction
    recall = 1 if not n_control else tp/n_control
    num = precision + recall
    f1= 0 if not num else 2*precision*recall/num
    return f1, precision, recall




