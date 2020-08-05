import pandas as pd, numpy as np, math
from pathlib import PurePath, Path
import re
import scipy
from scipy import io as dummy

def from_electro_gui(fname, fnames, fs):
    df = pd.read_excel(fname).rename(
        columns = {
            'Start (s)': 'start',
            'End (s)': 'end',
            'Label': 'label',
            'd_filenum': 'd_idx',
            'File #': 'f_idx',
            'Syllable #': 'syllable_idx'
        }
    )
    #df.f_idx = pd.Categorical(df.f_idx).codes
    #df['fname'] = df.f_idx.map(lambda i: fnames[i])
    #print(re.search('_d(\d+)_', df.fname[0]).group(1))
    df.start = (df.start*fs).astype(int)
    df.end = (df.end*fs).astype(int)

    fname_d_idx = list(map(lambda f : int(re.search('_d(\d+)_', f).group(1)), fnames))
    assert(len(fname_d_idx) == len(set(fname_d_idx)))

    df['fname'] = df.d_idx.map(lambda d : fnames[fname_d_idx.index(d)])
    df['dph'] = df.fname.map(lambda f : int(re.search('_dph(\d+)_', f).group(1)))
    df.drop(columns = ['f_idx', 'syllable_idx'], inplace = True)

    return df

def to_electro_gui(out_file, segments, fs):
    fnames = segments.fname.unique()
    n = len(fnames)
    segments.start += 1
    files = np.core.records.fromarrays([fnames], names=['name'])
    file_groups = segments.groupby('fname', as_index=False)
    scipy.io.savemat(out_file, {
        'dbase': {
            'PathName': 'wav_dir',
            'Times': [0] * n,
            'FileLength': [0] * n,
            'SoundFiles': files,
            'ChannelFiles': [files] + 6 * [{}],
            'SoundLoader': 'WaveRead',
            'ChannelLoader': ['WaveRead'] * 7,
            'Fs': fs,
            'SegmentThresholds': [math.inf] * n,
            'SegmentTimes': file_groups.apply(lambda g: [*zip(g.start.astype(float).values, g.end.values.astype(float))]).values,
            'SegmentTitles': file_groups.apply(lambda g: g.label.astype(str).values).values,
            'SegmentIsSelected': file_groups.apply(lambda g: [1] * len(g)).values,
            'EventSources': [],
            'EventFunctions': [],
            'EventDetectors': [],
            'EventThresholds': [],
            'EventTimes': [],
            'EventIsSelected': [],
            'Properties': {
                'Names': [[]]*n,
                'Values': [[]]*n,
                'Types': [[]]*n,
            },
            'AnalysisState': {
                'EventList': [],
                'SourceList': np.array(['(None)', 'Sound'], dtype='object'),
                'CurrentFile': 1,
                'EventWhichPlot': 0,
                'EventLims': [-1, 1]
            }
        }
    })


def to_raven(fname, segments, fs):
    df = pd.DataFrame()
    df['Begin Time'] = segments.start/fs
    df['End Time'] = segments.end/fs
    df['Low Frequency'] = 1
    df['High Frequency'] = 20000
    df.to_csv(fname, sep='\t', index = False)

def to_audacity(out_dir, segments, fs):
    df = pd.DataFrame()
    for fname in segments.fname.unique():
        s = segments[segments.fname == fname]
        df = pd.DataFrame({'start': s.start/fs, 'end': s.end/fs, 'label': s.label})
        df.to_csv(PurePath(out_dir).joinpath(PurePath(fname).name + '.audacity.txt'), sep='\t', index = False, header = False)

def from_audacity(fname, fs):
    s = pd.read_csv(fname, sep='\t', names=('start', 'end', 'label'))
    s = s[s.start != '\\']
    s.start = s.start.astype(float) * fs
    s.end *= fs
    return s

def to_avisoft(out_dir, segments, fs):
    df = pd.DataFrame()
    for fname in segments.fname.unique():
        s = segments[segments.fname == fname]
        df = pd.DataFrame({'start': s.start/fs, 'end': s.end/fs, 'label': s.label})
        df.to_csv(PurePath(out_dir).joinpath(PurePath(fname).name + '.avisoft.txt'), sep='\t', index = False, header = False, float_format='%.6f', line_terminator='\r\n')

def from_raven(fname, fs):
    s = pd.read_csv(fname, sep='\t').sort_values('Begin Time (s)')
    #fname_stem = PurePath(fname).name.split('.')[0] 
    #w['fname'] = fname_stem + '.wav'
    #w['whistle_idx'] = range(len(w))
    s=s[s.View == 'Waveform 1']
    s.rename(columns={
        'Begin Time (s)': 'start',
        'End Time (s)': 'end',
        'Low Freq (Hz)': 'freq_low',
        'High Freq (Hz)': 'freq_high'
    }, inplace=True)
    #whistles['whistle_bout_idx'] = whistles.Annotation.astype(str).map(lambda b: 1 if '.' in b else 0)
    s.start = (s.start*fs).astype(int)
    s.end = (s.end*fs).astype(int)
    s.drop(columns = ['View', 'Channel', 'Annotation', 'Selection'], inplace=True)
    return s
