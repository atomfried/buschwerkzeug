import numpy as np, pandas as pd, buschwerkzeug as bw
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Input, GRU, Dense, Activation, Dropout, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import Recall

import keras.backend as K
import sys
sys.setrecursionlimit(10000)


def get_model(data_in, data_out, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb, dropout_rate):

    spec_start = Input(shape=(data_in.shape[-3], data_in.shape[-2], data_in.shape[-1]))
    spec_x = spec_start
    for _i, _cnt in enumerate(_cnn_pool_size):
        spec_x = Conv2D(filters=_cnn_nb_filt, kernel_size=(3, 3), padding='same')(spec_x)
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)
        spec_x = MaxPooling2D(pool_size=(1, _cnn_pool_size[_i]))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)
    spec_x = Permute((2, 1, 3))(spec_x)
    spec_x = Reshape((data_in.shape[-2], -1))(spec_x)

    for _r in _rnn_nb:
        spec_x = Bidirectional(
            GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='mul')(spec_x)

    for _f in _fc_nb:
        spec_x = TimeDistributed(Dense(_f))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    spec_x = TimeDistributed(Dense(data_out.shape[-1]))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)

    _model = Model(inputs=spec_start, outputs=out)
    _model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', Recall()])
    #_model.summary()
    return _model

class CRNN(BaseEstimator):
    def __init__(self, seq_len, feature_hop_len, epochs, dropout_rate = 0.5, batch_size=128):
        self.seq_len = seq_len
        self.feature_hop_len = feature_hop_len
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size

    def fit(self, _X, _T):
        # pad all file features so sequences are split along files
        X = np.vstack(list(map(lambda x: np.pad(x, ((0, self.seq_len - x.shape[0] % self.seq_len), (0,0)), mode='constant'), _X)))
        T = np.zeros(len(X))
        offset = 0
        for features, segments in zip(_X, _T):
            for segment in segments.itertuples():
                start = int(np.floor(segment.start/self.feature_hop_len + offset))
                end = int(np.floor(segment.end/self.feature_hop_len + offset))
                T[start:end] = 1
            offset += len(features)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        #classes, class_counts = np.unique(T, return_counts=True)
        #class_weight = dict(zip(classes, map(lambda c: c/len(T), class_counts)))

        X = X.reshape(X.shape[0] // self.seq_len, 1, self.seq_len, X.shape[1])
        anti_T = np.array(list(map(lambda x:0 if x else 1, T)))
        anti_T = anti_T.reshape(anti_T.shape[0] // self.seq_len, self.seq_len, 1)
        T = T.reshape(T.shape[0] // self.seq_len, self.seq_len, 1)
        T = np.concatenate((T,anti_T),axis=2)

        cnn_nb_filt = 128            # CNN filter size
        cnn_pool_size = [5, 2, 2]   # Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
        rnn_nb = [32, 32]           # Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
        fc_nb = [32]                # Number of FC nodes.  Length of fc_nb =  number of FC layers
        dropout_rate = 0.1          # Dropout after each layer

        self.clf = get_model(X, T, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, self.dropout_rate)
        print('learning rate:', K.eval(self.clf.optimizer.lr))
        self.clf.fit(X, T, epochs=self.epochs, batch_size=self.batch_size) #, class_weight=class_weight) #, verbose=0)
        return self

    def predict(self, X):
        X = list(map(lambda x: np.pad(x, ((0, self.seq_len - x.shape[0] % self.seq_len), (0,0)), mode='constant'), X))
        lens = map(len, X)
        X = np.vstack(X)
        X = self.scaler.transform(X)
        X = X.reshape(X.shape[0] // self.seq_len, 1, self.seq_len, X.shape[1])
        _Y = self.clf.predict(X)
        _Y = _Y.flatten()
        th = 0.5
        print('>= th:' + str((_Y >= th).sum()))
        print('distribution:', np.mean(_Y), np.std(_Y))
        Y = []
        start = 0
        hold_len = 0.03 * 40000 / self.feature_hop_len
        for l in lens:
            Y.append(pd.DataFrame(bw.segments.detect(_Y[start:start + l], hold_len, th), columns=('start', 'end')))
            start += l
        return Y

