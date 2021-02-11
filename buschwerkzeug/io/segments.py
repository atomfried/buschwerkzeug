import pandas as pd, numpy as np, math
from pathlib import PurePath, Path
import re
import scipy
from scipy import io as dummy

def from_electro_gui_linda(fname, fnames, fs):
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

def from_electro_gui(fname):
    db = scipy.io.loadmat(fname)['dbase']
    fnames = [x[0][0][0] for x in db['SoundFiles'][0][0]]
    segments = []
    for i in range(len(fnames)):
        segments.append(pd.DataFrame.from_records({
            'start': [ x[0] for x in db['SegmentTimes'][0][0][0][i]],
            'end': [ x[1] for x in db['SegmentTimes'][0][0][0][i]],
            'label': [ x[0] if x else None for x in db['SegmentTitles'][0][0][0][i][0] ]
            }))
    return pd.concat(segments, keys=fnames, names=['fname']).reset_index()
    #return pd.Series(segments, fnames)  


def to_electro_gui(out_file, segments, fs):
    #fnames = segments.fname.unique()
    segments.start += 1
    file_groups = segments.groupby('fname', as_index='False')
    fnames = file_groups.first().index.values
    #print(fnames)
    n = len(fnames)
    fnames = np.core.records.fromarrays([fnames], names=['name'])
    scipy.io.savemat(out_file, {
        'dbase': {
            'PathName': 'wav_dir',
            'Times': [0] * n,
            'FileLength': [0] * n,
            'SoundFiles': fnames,
            'ChannelFiles': [fnames] + 6 * [{}],
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

import warnings
def from_raven(fname, fs):
    s = pd.read_csv(fname, sep='\t').sort_values('Begin Time (s)')
    #fname_stem = PurePath(fname).name.split('.')[0] 
    #w['fname'] = fname_stem + '.wav'
    #w['whistle_idx'] = range(len(w))
    if 'Annotation' not in s:
        warnings.warn('{}: No "Annotation" column. Using column named "{}"'.format(fname, s.columns[7]))
        s.rename(columns = { s.columns[7]: 'Annotation'}, inplace = True)
    s=s[s.View == 'Waveform 1']
    s.rename(columns={
        'Begin Time (s)': 'start',
        'End Time (s)': 'end',
        'Low Freq (Hz)': 'freq_low',
        'High Freq (Hz)': 'freq_high',
        'Annotation': 'label'
    }, inplace=True)
    #whistles['whistle_bout_idx'] = whistles.Annotation.astype(str).map(lambda b: 1 if '.' in b else 0)
    s.start = (s.start*fs).astype(int)
    s.end = (s.end*fs).astype(int)
    s.drop(columns = ['View', 'Channel', 'Selection'], inplace=True)
    return s.reset_index(drop=True)

def to_raven(out_dir, segments, fs):
    for fname in segments.fname.unique():
        s = segments[segments.fname == fname]
        df = pd.DataFrame()
        df['Selection'] = range(1, len(s)+1)
        df['View'] = 'Spectrogram 1'
        df['Channel'] = 1
        df['Begin Time (s)'] = (s.start/fs).values
        df['End Time (s)'] = (s.end/fs).values
        df['Low Frequency (Hz)'] = (s.freq_low).values
        df['High Frequency (Hz)'] = (s.freq_high).values
        df['Annotation'] = range(len(s))
        df.to_csv(PurePath(out_dir).joinpath(PurePath(fname).stem+'.raven.txt'), sep='\t', index = False)

def from_audacity(fname, fs):
    s = pd.read_csv(fname, sep='\t', names=('start', 'end', 'label'))
    s = s[s.start != '\\']
    s.start = s.start.astype(float) * fs
    s.end *= fs
    return s

def to_audacity(out_dir, segments, fs):
    for fname in segments.fname.unique():
        s = segments[segments.fname == fname]
        df = pd.DataFrame({'start': s.start/fs, 'end': s.end/fs, 'label': s.label})
        df.to_csv(PurePath(out_dir).joinpath(PurePath(fname).stem + '.audacity.txt'), sep='\t', index = False, header = False)


def to_avisoft(out_dir, segments, fs):
    for fname in segments.fname.unique():
        s = segments[segments.fname == fname]
        df = pd.DataFrame({'start': s.start/fs, 'end': s.end/fs, 'label': s.label})
        df.to_csv(PurePath(out_dir).joinpath(PurePath(fname).stem + '.avisoft.tsv'), sep='\t', index = False, header = False, float_format='%.6f', line_terminator='\r\n')

def to_textgrid(out_dir, segments, fs):
    for fname in segments.fname.unique():
        _segments = segments[segments.fname == fname]
        last = _segments.end.max()/fs
        with open(PurePath(out_dir).joinpath(PurePath(fname).stem + '.TextGrid'), 'w') as f:
            f.write((
                'File type = "ooTextFile"\n'
                'Object class = "TextGrid"\n\n'
                'xmin = 0\n'
                'xmax = {:.2f}\n'
                'tiers? <exists>\n'
                'size = 1\n'
                'item []\n'
                'item [1]:\n'
                'class = "IntervalTier"\n'
                'name = "tier name"\n'
                'xmin = 0\n'
                'xmax = {:.2f}\n'
                'intervals: size = {}\n')
            .format(last, last, len(_segments)))
            i=0
            for segment in _segments.itertuples():
                i+=1
                f.write('intervals [{}]:\nxmin = {:.2f}\nxmax = {:.2f}\ntext = "{}"\n'.format(i, segment.start/fs, segment.end/fs, segment.label))

