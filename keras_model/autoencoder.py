import copy
import numpy as np

import sys, os, pickle, json

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Lambda, Layer, BatchNormalization, Dropout
from keras import metrics
from keras import backend as K
from keras import losses
from keras.regularizers import l1_l2

from sklearn.metrics import mean_squared_error


def instance_wise_model_mse(model, X):
    X_recon = model.predict(X)
    return np.array([mean_squared_error(true, recon) for true, recon in zip(X, X_recon)])

class Autoencoder(object):
    """
    Autoencoder
    
    """

    def __init__(self, input_dim, hidden_dims, decoder_hidden_dims=None, 
                 activation='relu', last_activation='linear',
                  batch_norm=False, dropout_prob=None, denoising_prob=None, 
                  reg_weight=0.0, reg1_weight=0.0):
        self._input_dim = input_dim
        self._encoder_hidden_dims = hidden_dims[:-1]
        self._latent_dim = hidden_dims[-1]
        self._decoder_hidden_dims = decoder_hidden_dims if decoder_hidden_dims is not None else self._encoder_hidden_dims[::-1]
        self._activation = activation
        self._last_activation = last_activation
        self._batch_norm = batch_norm
        self._dropout_prob = dropout_prob
        self._denoising_prob = denoising_prob
        self._reg1_weight = reg1_weight
        self._reg2_weight = reg_weight

        self._model = None
        self._encoder = None
        self._decoder = None

        self._construct_network()

    def _construct_network(self):
        input = Input(shape=(self._input_dim,))

        encoded = input
        if self._denoising_prob is not None and self._denoising_prob > 0:
            encoded = Dropout(self._denoising_prob)(encoded)
        for h_dim in self._encoder_hidden_dims + [self._latent_dim]:
            encoded = Dense(h_dim, 
                            activation=self._activation, 
                            kernel_regularizer=l1_l2(self._reg1_weight, self._reg2_weight))(encoded)

            if self._batch_norm:
                encoded = BatchNormalization()(encoded)
            if self._dropout_prob is not None and self._dropout_prob > 0:
                encoded = Dropout(self._dropout_prob)(encoded)

        # latent variable for given data
        z = encoded

        # latent variable directly given
        input_z = Input(shape=(self._latent_dim,))

        decoded = [z, input_z]
        for h_dim in self._decoder_hidden_dims:
            decoding_layer = Dense(h_dim, 
                                   activation=self._activation, 
                                   kernel_regularizer=l1_l2(self._reg1_weight, self._reg2_weight))
            decoded = [decoding_layer(each_tensor) for each_tensor in decoded]

            if self._batch_norm:
                batch_layer = BatchNormalization()
                decoded = [batch_layer(each_tensor) for each_tensor in decoded]
            if self._dropout_prob is not None and self._dropout_prob > 0:
                dropout_layer = Dropout(self._dropout_prob)
                decoded = [dropout_layer(each_tensor) for each_tensor in decoded]

        last_layer = Dense(self._input_dim, 
                           activation=self._last_activation,
                           kernel_regularizer=l1_l2(self._reg1_weight, self._reg2_weight))
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

    def fit(self, X, pretrain=None, **kwargs):
        if pretrain is not None:
            layer_sizes = [l.output_shape for l in self._model.layers]
            is_symm = np.array([f == b for f, b in zip(layer_sizes, layer_sizes[::-1])]).all()
            if not is_symm:
                raise Exception('pretrain for asymmetric network is not implemented')

            pre_kwargs = copy.deepcopy(kwargs)
            pre_kwargs['epochs'] = pretrain['epochs']

            encoder_height = len(self._encoder.layers)
            decoder_height = len(self._decoder.layers)
            for i in range(2, encoder_height):
                target_layers = self._encoder.layers[:i] + self._decoder.layers[(decoder_height-i+1):]
                pre_ae = Sequential(target_layers)
                pre_ae.compile(**pretrain['compile_args'])
                pre_ae.fit(X, X, **pre_kwargs)

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


class VariationalAutoencoder(object):
    AVAILABLE_RECON_LOSSES = ['mse', 'mean_squared_error']

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

        return K.mean(kl_loss + recon_loss)

    def _construct_model(self):
        # inputs for data and random
        input_x = Input(shape=(self._input_dim,))
        input_z = Input(shape=(self._latent_dim,))

        # encoding
        encoded = input_x
        if self._denoising_prob is not None and self._denoising_prob > 0:
            encoded = Dropout(self._denoising_prob)(encoded)
        for h_dim in self._encoder_hidden_dims + [self._latent_dim]:
            encoding_layer = Dense(h_dim, activation=self._activation)
            encoded = encoding_layer(encoded)
            if self._batch_norm:
                encoded = BatchNormalization()(encoded)
            if self._dropout_prob is not None and self._dropout_prob > 0:
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
            if self._dropout_prob is not None and self._dropout_prob > 0:
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
    """
    Embedding is guided by the number of clusters given by a parameter
    """
    def __init__(self, n_clusters, cls_loss_weight=1.0, **kwargs):
        super(GuidedAutoencoder, self).__init__(**kwargs)
        self._n_clusters = n_clusters
        self._cls_loss_weight = cls_loss_weight
        self._loss = None

        borders = np.round(np.linspace(0, self._latent_dim, self._n_clusters+1)).astype('int')
        cluster_centers = []
        for i, (left, right) in enumerate(zip(borders[:-1], borders[1:])):
            cls_center = np.zeros(self._latent_dim)
            cls_center[left:right] = 1.0 * (2*(i%2)-1)
            cluster_centers.append(cls_center)
        self._cluster_centers = np.array(cluster_centers)

    def _gae_loss(self, x, x_recon):
        recon_loss_func = losses.get(self._loss)
        recon_loss = K.mean(recon_loss_func(x, x_recon))

        cls_loss = 0
        for mu in self._cluster_centers:
            cls_loss += K.mean(metrics.mean_squared_error(self._z, mu))
        cls_loss /= self._n_clusters

        return recon_loss - self._cls_loss_weight*cls_loss

    def compile(self, **kwargs):
        self._loss = kwargs.get('loss', 'mse')
        kwargs['loss'] = self._gae_loss

        self._model.compile(**kwargs)


#################################################################################
# Wrapper for dr. yoon's implementation
#################################################################################
#import sys
#sys.path.append('autoencoder_yoon')
#from ae import AE
#from vae import VAE
#
#class AE_Wrapper(object):
#    def __init__(self, **kwargs):
#        self.ae_obj = AE(**kwargs)
#        self.model = self.ae_obj.model()
#
#    def fit(self, X, **kwargs):
#        return self.model.fit(X, X, **kwargs)
#
#    def compile(self, **kwargs):
#        self.model.compile(**kwargs)
#
#    def summary(self, enter=0):
#        self.model.summary()
#        print ''.join(['\n' for _ in range(enter)])
#
#    def predict(self, X):
#        return self.model.predict(X)
#
#    def encode(self, X):
#        return self.ae_obj.encoder().predict(X)
#
#    def decision_function(self, X):
#        return -instance_wise_model_mse(self.model, X)
#
#
#class VAE_Wrapper(AE_Wrapper):
#    def __init__(self, **kwargs):
#        self.ae_obj = VAE(**kwargs)
#        self.model = self.ae_obj.model()
#        self.vae_loss = self.ae_obj.loss
#
#    def compile(self, **kwargs):
#        kwargs['loss'] = self.vae_loss
#        self.model.compile(**kwargs)
#
#    def decision_function(self, X):
#        return -instance_wise_model_mse(self.model, X)





