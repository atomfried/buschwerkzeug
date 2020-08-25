import numpy as np, soundfile as sf, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from pathlib import PurePath, Path
from joblib import Memory


class Detector(BaseEstimator):
    def __init__(self, segmenters, build_features, clf, tolerance, cache_dir = '/tmp/segment_feature_cache', scale =True):
        self.segmenters = segmenters
        self.build_features = build_features
        self.clf = clf
        self.tolerance= tolerance
        self.scale = scale
        self.scaler = StandardScaler()
        self.cache_dir=cache_dir

    def make_cached_f(self, segmenter, cache_id, wav_dir):
        cache_dir = str(PurePath(self.cache_dir).joinpath(cache_id))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        def f(fname):
            candidates = segmenter(str(PurePath(wav_dir).joinpath(fname)))
            candidates['fname'] = fname
            features = self.build_features(candidates, wav_dir)
            return candidates, features
        return Memory(cache_dir).cache(f)

    def fit(self, fnames, fsegments, wav_dir = '', candidate_table = None):
        candidates_features_builders = [ self.make_cached_f(self.segmenters[segmenter_id], segmenter_id, wav_dir) for segmenter_id in self.segmenters]
        candidates = pd.DataFrame()
        features = []
        missed = []
        i = 0
        for fname, segments in zip(fnames, fsegments):
            i += 1
            #print('fit -- {}/{}: {}'.format(i, len(fnames), fname))
            for build_candidates_features in candidates_features_builders:
                _candidates, _features = build_candidates_features(fname)
                _candidates['label'] = 0.0
                for segment in segments.itertuples():
                    _candidates.loc[
                        ((_candidates.start-segment.start).abs()<=self.tolerance) &
                        ((_candidates.end-segment.end).abs()<=self.tolerance),
                        'label'] = 1.0
                candidates = candidates.append(_candidates)
                features.extend(_features)
        if candidate_table:
            candidates.to_csv(candidate_table)
        X = np.array(features)
        if self.scale:
            X = self.scaler.fit_transform(X)
        T = candidates.label
        self.clf.fit(X,T)
        return self

    def predict(self, fnames, wav_dir =''):
        if len(self.clf.classes_) != 2:
            assert('classifier' == 'useless')
            #return [ pd.DataFrame(columns=('start', 'end')) ] * len(fnames)
        candidates_features_builders = [ self.make_cached_f(self.segmenters[segmenter_id], segmenter_id, wav_dir) for segmenter_id in self.segmenters]
        candidates = pd.DataFrame()
        features = []
        i = 0
        for fname in fnames:
            i += 1
            #print('predict -- {}/{}: {}'.format(i, len(fnames), fname))
            for build_candidates_features in candidates_features_builders:
                _candidates, _features = build_candidates_features(fname)
                candidates = candidates.append(_candidates)
                features.extend(_features)
        X = np.array(features)
        if self.scale:
            X = self.scaler.transform(features)
        candidates['prob'] = self.clf.predict_proba(X)[:,1]
        #candidates['prob0'] = self.clf.predict_proba(X)[:,0]
        if True:
            self.predict_candidates = candidates #[candidates[candidates.fname == fname] for fname in fnames]
        candidates = candidates[candidates.prob > 0.5]
        candidates['selected'] = True
        for candidate in candidates.itertuples():
            candidates.loc[
                (candidates.fname == candidate.fname) &
                (candidates.start < candidate.end) &
                (candidates.end > candidate.start) &
                (candidates.prob <= candidate.prob) &
                (candidates.index > candidate.Index),
                'selected'
            ]= False
        candidates = candidates[candidates.selected]
        return list(map(lambda f : candidates[candidates.fname == f], fnames))

