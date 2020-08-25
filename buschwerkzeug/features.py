import numpy as np, pandas as pd, soundfile as sf
import skimage.filters as dummy
import skimage
from . import signal
from librosa import feature
from .vggish.vggish import VGGish
from functools import lru_cache
from pathlib import PurePath


@lru_cache(maxsize=1)
def load_wav(fname):
    return sf.read(fname)

def spectrogram_builder(win_len, max_len):
    def build_spectrograms(segments, wav_dir):
        S = []
        for segment in segments.itertuples():
            wav, fs = load_wav(PurePath(wav_dir).joinpath(segment.fname))
            wav = wav[segment.start:segment.end]
            assert(len(wav) <= max_len)
            pad_len = max_len - len(wav)
            l_pad = pad_len // 2
            r_pad = pad_len - l_pad
            wav = np.pad(wav, ((l_pad,r_pad)))
            f,t,_S = signal.spectrogram(wav, fs, win_len)
            S.append(_S[:,:,np.newaxis])
        return S
    return build_spectrograms


def all_equal(l):
    first = l[0]
    return all(x==first for x in l)

def feature_builder(vggish_model, vggish_pca_params, normalize=False):
    return lambda segments, wav_dir: features(segments, wav_dir, vggish_model, vggish_pca_params, normalize).values

def features(segments, wav_dir, vggish_model, vggish_pca_params, normalize=False):
    shape_features = pd.DataFrame()
    rosa_features = pd.DataFrame()
    vggish_features = []

    vggish = VGGish(vggish_model, vggish_pca_params)

    i = 0
    for segment in segments.itertuples():
        i+=1
        print('Building features ({}/{})'.format(i, len(segments)))

        fname = str(PurePath(wav_dir).joinpath(segment.fname))
        wav,fs = load_wav(fname)
        wav = wav[int(segment.start):int(segment.end)]
        if normalize:
            if normalize == 'rms':
                wav = signal.rms_normalize(wav)
            else:
                assert('unknown normalize method')

        vggish_features.append(vggish.features(wav, fs))

        rosa_features = rosa_features.append({
            'mean_chroma': np.mean(feature.chroma_stft(wav, fs)),
            'spectral_centroid': np.mean(feature.spectral_centroid(wav,fs)),
            'spectral_bandwidth': np.mean(feature.spectral_bandwidth(wav, fs)),
            'spectral_rolloff': np.mean(feature.spectral_rolloff(wav,fs)),
            'rms': np.mean(feature.rms(wav)),
            'zero_crossing_rate': np.mean(feature.zero_crossing_rate(wav))
        }, ignore_index=True)

        f,t,S = signal.spectrogram(wav, fs)
        th = -float('inf') if all_equal(S.flatten()) else skimage.filters.threshold_otsu(S)
        mask = (S>th)*1
        lprops = skimage.measure.regionprops(mask, S)
        assert(len(lprops)==1)
        props = lprops[0]
        shape_features = shape_features.append({
            'onset': segment.start,
            'offset': segment.end,
            'duration': segment.end-segment.start,
            'area': props.area,
            'centroid0': props.centroid[0],
            'centroid1': props.centroid[1],
            'convex_area': props.convex_area,
            'eccentricity': props.eccentricity,
            'filled_area': props.filled_area,
            'equivalent_diameter': props.equivalent_diameter,
            'extent': props.extent,
            'inertia_tensor00': props.inertia_tensor[0][0],
            'inertia_tensor01': props.inertia_tensor[0][1],
            'inertia_tensor10': props.inertia_tensor[1][0],
            'inertia_tensor11': props.inertia_tensor[1][1],
            'inertia_tensor_eigval0': props.inertia_tensor_eigvals[0],
            'inertia_tensor_eigval1': props.inertia_tensor_eigvals[1],
            'centroid_x': props.local_centroid[1],
            'centroid_y': props.local_centroid[0],
            'major_axis_length': props.major_axis_length,
            'minor_axis_length': props.minor_axis_length,
            'min_intensity': props.min_intensity,
            'mean_intensity': props.mean_intensity,
            'max_intensity': props.max_intensity,
            'orientation': props.orientation,
            'solidity': props.solidity,
            'weighted_centroid0': props.weighted_local_centroid[0],
            'weighted_centroid1': props.weighted_local_centroid[1]
            #TODO: moments
            }, ignore_index=True)

    return pd.concat([
        rosa_features.add_prefix('rosa_'),
        shape_features.add_prefix('shape_'),
        pd.DataFrame(vggish_features).add_prefix('vggish_')
    ], axis=1)
