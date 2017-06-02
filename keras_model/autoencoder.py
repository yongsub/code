import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda, Layer, BatchNormalization, Dropout
from keras import metrics
from keras import backend as K

from sklearn.metrics import mean_squared_error


def instance_wise_model_mse(model, X):
    X_recon = model.predict(X)
    return np.array([mean_squared_error(true, recon) for true, recon in zip(X, X_recon)])

class Autoencoder:
    """
    Autoencoder
    
    """

    def __init__(self, input_dim, hidden_dims, decoder_hidden_dims=None, activation='relu', last_activation='linear', batch_norm=False, dropout_prob=None, denoising_prob=None):
        self._input_dim = input_dim
        self._encoder_hidden_dims = hidden_dims[:-1]
        self._latent_dim = hidden_dims[-1]
        self._decoder_hidden_dims = decoder_hidden_dims if decoder_hidden_dims is not None else self._encoder_hidden_dims[::-1]
        self._activation = activation
        self._last_activation = last_activation
        self._batch_norm = batch_norm
        self._dropout_prob = dropout_prob
        self._denoising_prob = denoising_prob

        self._model = None
        self._encoder = None
        self._decoder = None

        self._construct_network()

    def _construct_network(self):
        input = Input(shape=(self._input_dim,))

        encoded = input
        if self._denoising_prob is not None:
            encoded = Dropout(self._denoising_prob)(encoded)
        for h_dim in self._encoder_hidden_dims:
            encoded = Dense(h_dim, activation=self._activation)(encoded)
            if self._batch_norm:
                encoded = BatchNormalization()(encoded)
            if self._dropout_prob is not None:
                encoded = Dropout(self._dropout_prob)(encoded)

        # latent variable calculated from data
        z = Dense(self._latent_dim, activation=self._activation)(encoded)

        # latent variable directly given
        input_z = Input(shape=(self._latent_dim,))

        decoded = [z, input_z]
        for h_dim in self._decoder_hidden_dims:
            decoding_layer = Dense(h_dim, activation=self._activation)
            decoded = [decoding_layer(each_tensor) for each_tensor in decoded]
            if self._batch_norm:
                batch_layer = BatchNormalization()
                decoded = [batch_layer(each_tensor) for each_tensor in decoded]
            if self._dropout_prob is not None:
                dropout_layer = Dropout(self._dropout_prob)
                decoded = [dropout_layer(each_tensor) for each_tensor in decoded]

        last_layer = Dense(self._input_dim, activation=self._last_activation)
        decoded = [last_layer(each_tensor) for each_tensor in decoded]

        self._encoder = Model(input, z)
        self._decoder = Model(input_z, decoded[1])
        self._model = Model(input, decoded[0])

    def summary(self, enter=0):
        line_breaks = ''.join(['\n' for i in range(enter)])

        self._model.summary()
        print line_breaks

    def compile(self, **kwargs):
        self._model.compile(**kwargs)

    def fit(self, X, **kwargs):
        return self._model.fit(X, X, **kwargs)

    def predict(self, X):
        N = X.shape[0]
        return self._model.predict(X).reshape(N, -1)

    def encode(self, X):
        return self._encoder.predict(X)

    def decode(self, Z):
        return self._decoder.predict(Z)

    def instance_wise_mse(self, X):
        return instance_wise_model_mse(self._model, X)

    def decision_function(self, X):
        return -self.instance_wise_mse(X)


class VariationalAutoencoder:
    AVAILABLE_RECON_LOSSES = ['mse', 'mean_squared_error']

    def __init__(self, input_dim, hidden_dims, decoder_hidden_dims=None, activation='relu', last_activation='linear', batch_norm=False, dropout_prob=None, denoising_prob=None, recon_weight=1.0):
        self._input_dim = input_dim
        self._encoder_hidden_dims = hidden_dims[:-1]
        self._latent_dim = hidden_dims[-1]
        self._decoder_hidden_dims = decoder_hidden_dims if decoder_hidden_dims is not None else self._encoder_hidden_dims[::-1]
        self._activation = activation
        self._last_activation = last_activation
        self._batch_norm = batch_norm
        self._dropout_prob = dropout_prob
        self._denoising_prob = denoising_prob
        self._recon_weight = recon_weight
        self._loss = None

        self._construct_model()

    def _sample(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def _vae_loss(self, x, x_recon):
        if self._loss in ['mse', 'mean_squared_error']:
            recon_loss = self._input_dim * metrics.mean_squared_error(x, x_recon)
        elif self._loss in ['binary_crossentropy']:
            recon_loss = self._input_dim * metrics.binary_crossentropy(x, x_recon)
        else:
            assert False, 'unsupported loss={} in _vae_loss'.format(self._loss)

        kl_loss = - 0.5 * K.sum(1 + self._z_log_var - K.square(self._z_mean) - K.exp(self._z_log_var), axis=-1)

        return K.mean(kl_loss + self._recon_weight*recon_loss)

    def _construct_model(self):
        # inputs for data and random
        input_x = Input(shape=(self._input_dim,))
        input_z = Input(shape=(self._latent_dim,))

        # encoding
        encoded = input_x
        if self._denoising_prob is not None:
            encoded = Dropout(self._denoising_prob)(encoded)
        for h_dim in self._encoder_hidden_dims + [self._latent_dim]:
            encoding_layer = Dense(h_dim, activation=self._activation)
            encoded = encoding_layer(encoded)
            if self._batch_norm:
                encoded = BatchNormalization()(encoded)
            if self._dropout_prob is not None:
                encoded = Dropout(self._dropout_prob)(encoded)

        self._z_mean = Dense(self._latent_dim, activation='linear')(encoded)
        self._z_log_var = Dense(self._latent_dim, activation='linear')(encoded)
        z = Lambda(self._sample, output_shape=(self._latent_dim,))([self._z_mean, self._z_log_var])

        # decoding
        decoded = [z, input_z]
        for h_dim in self._decoder_hidden_dims:
            decoding_layer = Dense(h_dim, activation=self._activation)
            decoded = [decoding_layer(each_tensor) for each_tensor in decoded]
            if self._batch_norm:
                batch_layer = BatchNormalization()
                decoded = [batch_layer(each_tensor) for each_tensor in decoded]
            if self._dropout_prob is not None:
                dropout_layer = Dropout(self._dropout_prob)
                decoded = [dropout_layer(each_tensor) for each_tensor in decoded]

        last_layer = Dense(self._input_dim, activation=self._last_activation)
        decoded = [last_layer(each_tensor) for each_tensor in decoded]

        # construct models
        self._encoder = Model(input_x, self._z_mean)
        self._encoder_rand = Model(input_x, z)
        self._decoder = Model(input_z, decoded[1])
        self._model = Model(input_x, decoded[0])

        self._encoder_log_var = Model(input_x, self._z_log_var)

    def compile(self, **kwargs):
        self._loss = kwargs.get('loss', 'mse')
        if self._loss not in self.AVAILABLE_RECON_LOSSES:
            raise Exception('{} are available for loss'.format(self.AVAILABLE_RECON_LOSSES))

        kwargs['loss'] = self._vae_loss

        self._model.compile(**kwargs)

    def fit(self, X, **kwargs):
        return self._model.fit(X, X, **kwargs)

    def predict(self, X):
        return self._model.predict(X)

    def encode(self, X, rand=False):
        if rand:
            return self._encoder_rand.predict(X)
        else:
            return self._encoder.predict(X)

    def decode(self, Z):
        return self._decoder.predict(Z)

    def generate(self, n):
        Z = K.random_normal(shape=(n, self._latent_dim), mean=0.0, stddev=1.0)
        return self.decode(Z)

    def instance_wise_mse(self, X):
        return instance_wise_model_mse(self._model, X)

    def decision_function(self, X):
        return -self.instance_wise_mse(X)

    def summary(self, all=False):
        if all:
            print 'Overall Model:'
            self.model_summary()
            print

            print 'Encoder:'
            self.encoder_summary()
            print

            print 'Encoder (Random):'
            self.encoder_rand_summary()
            print

            print 'Decoder:'
            self.decoder_summary()
        else:
            self.model_summary()

    def model_summary(self):
        self._model.summary()
    def encoder_summary(self):
        self._encoder.summary()
    def encoder_rand_summary(self):
        self._encoder_rand.summary()
    def decoder_summary(self):
        self._decoder.summary()


class GuidedAutoencoder(Autoencoder):
    def __init__(self, knn, knn_weight, **kwargs):
        super(GuidedAutoencoder, self).__init__(**kwargs)
        self._knn = knn
        self._knn_kernel = None
        self._knn_weight = knn_weight
        self._loss = None

        self._construct_network()

    def _gae_loss(self, x, x_recon):
        if self._loss in ['mse', 'mean_squared_error']:
            recon_loss = self._input_dim * metrics.mean_squared_error(x, x_recon)
        elif self._loss in ['binary_crossentropy']:
            recon_loss = self._input_dim * metrics.binary_crossentropy(x, x_recon)
        else:
            assert False, 'unsupported loss={} in _vae_loss'.format(self._loss)

        loc_loss = z;


        return K.mean(kl_loss + recon_loss)

    def _construct_network(self):
        input = Input(shape=(self._input_dim,))

        encoded = input
        if self._denoising_prob is not None:
            encoded = Dropout(self._denoising_prob)(encoded)
        for h_dim in self._encoder_hidden_dims:
            encoded = Dense(h_dim, activation=self._activation)(encoded)
            if self._batch_norm:
                encoded = BatchNormalization()(encoded)
            if self._dropout_prob is not None:
                encoded = Dropout(self._dropout_prob)(encoded)

        # latent variable calculated from data
        z = Dense(self._latent_dim, activation=self._activation)(encoded)

        # latent variable directly given
        input_z = Input(shape=(self._latent_dim,))

        decoded = [z, input_z]
        for h_dim in self._decoder_hidden_dims:
            decoding_layer = Dense(h_dim, activation=self._activation)
            decoded = [decoding_layer(each_tensor) for each_tensor in decoded]
            if self._batch_norm:
                batch_layer = BatchNormalization()
                decoded = [batch_layer(each_tensor) for each_tensor in decoded]
            if self._dropout_prob is not None:
                dropout_layer = Dropout(self._dropout_prob)
                decoded = [dropout_layer(each_tensor) for each_tensor in decoded]

        last_layer = Dense(self._input_dim, activation=self._last_activation)
        decoded = [last_layer(each_tensor) for each_tensor in decoded]

        self._encoder = Model(input, encoded)
        self._decoder = Model(input_z, decoded[1])
        self._model = Model(input, decoded[0])


#################################################################################
# Wrapper for dr. yoon's implementation
#################################################################################
import sys
sys.path.append('autoencoder_yoon')
from ae import AE
from vae import VAE

class AE_Wrapper:
    def __init__(self, **kwargs):
        self.ae_obj = AE(**kwargs)
        self.model = self.ae_obj.model()

    def fit(self, X, **kwargs):
        return self.model.fit(X, X, **kwargs)

    def compile(self, **kwargs):
        self.model.compile(**kwargs)

    def summary(self, enter=0):
        self.model.summary()
        print ''.join(['\n' for _ in range(enter)])

    def predict(self, X):
        return self.model.predict(X)

    def encode(self, X):
        return self.ae_obj.encoder().predict(X)

    def decision_function(self, X):
        return -instance_wise_model_mse(self.model, X)


class VAE_Wrapper(AE_Wrapper):
    def __init__(self, **kwargs):
        self.ae_obj = VAE(**kwargs)
        self.model = self.ae_obj.model()
        self.vae_loss = self.ae_obj.loss

    def compile(self, **kwargs):
        kwargs['loss'] = self.vae_loss
        self.model.compile(**kwargs)

    def decision_function(self, X):
        return -instance_wise_model_mse(self.model, X)





