from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout, Flatten, GRU, Input, Merge
from keras.layers.core import Lambda, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import Model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
import numpy as np
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers


def root_mean_squared_log_error(y_true, y_pred):
    y_pred = K.clip(y_pred, 0, 1e10)
    rmsle = K.sqrt(K.mean(
        K.pow(K.log(y_true + 1) - K.log(y_pred + 1), 2)
        ))

    return rmsle


class DNNRegressor:

    def _build_model(self, input_dim):
        self.model = Sequential()
        self.model.add(Dense(1024, input_shape=(input_dim,)))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(512))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(1))
        self.model.summary()

        optimizer = Adam(lr=self.lr, decay=self.lr_decay)

        self.model.compile(loss=root_mean_squared_log_error,
                           optimizer=optimizer,
                           metrics=[root_mean_squared_log_error])

    def __init__(self, n_iters=1000, lr=0.1,
                 lr_decay=0.01, batch_size=256,
                 filename='best.h5', ram=0.2, valid=None):
        self.n_iters = n_iters
        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.model = None
        self.filename = filename
        self.valid = valid

        # set GPU memory limit
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = ram
        set_session(tf.Session(config=config))

    def fit(self, X, y, valid=None):
        if self.model is None:
            self._build_model(X.shape[1])

        if self.valid is not None:
            valid = (self.valid['x'], self.valid['y'])

        # earlystopping = EarlyStopping(monitor='val_root_mean_squared_log_error',
        #                               patience=15,
        #                               mode='min')

        checkpoint = ModelCheckpoint(filepath=self.filename,
                                     verbose=1,
                                     save_best_only=True,
                                     monitor='val_root_mean_squared_log_error',
                                     mode='min')

        self.model.fit(X, y,
                       epochs=self.n_iters,
                       validation_data=valid,
                       batch_size=self.batch_size,
                       callbacks=[checkpoint])

    def load(self, filename):
        self.model = load_model(filename)

    def predict_raw(self, X):
        return self.model.predict(X)

    def predict(self, X, threshold=0.5):
        predict = self.model.predict(X)
        return np.where(predict > threshold, 1, 0)
