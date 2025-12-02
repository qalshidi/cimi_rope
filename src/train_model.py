"""Code to train the Autoencoder.

    Usage:

"""

__author__ = 'Qusai Al Shidi'
__email__ = 'qusai.alshidi@mail.wvu.edu'

from functools import partial
from copy import deepcopy
import sys
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import keras_tuner
import keras
from keras import layers, ops, regularizers
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Orthogonal factor (ZTZ-I)^2
@tf.function
def orthogonal_factor(tensor):
    shape = tensor.shape[0]
    a = tf.reshape(tensor, (shape, 1))
    return tf.norm(tf.matmul(a, tf.transpose(a)) - tf.eye(shape),
                   axis=(0, 1))**2


def create_oae_model(hyperparameters=None):
    """

    Args:

    Returns:

    Raises:

    Examples:
    """
    optimizer_e = keras.optimizers.Adam(weight_decay=1e-7)
    optimizer_d = keras.optimizers.Adam(weight_decay=1e-7)
    # optimizer = hyperparameters.Choice('optimizer',
    #     [keras.optimizers.Adam(),
    #      keras.optimizers.SGD(),
    #      keras.optimizers.RMSprop()])

    oae = OrthogonalAutoEncoder(hyperparameters)
    oae.compile(optimizer_e=optimizer_e, optimizer_d=optimizer_d)
    return oae


def create_coae_model(hyperparameters=None):
    """

    Args:

    Returns:

    Raises:

    Examples:
    """
    optimizer_e = keras.optimizers.Adam(weight_decay=1e-7)
    optimizer_d = keras.optimizers.Adam(weight_decay=1e-7)
    # optimizer = hyperparameters.Choice('optimizer',
    #     [keras.optimizers.Adam(),
    #      keras.optimizers.SGD(),
    #      keras.optimizers.RMSprop()])

    coae = COAE(hyperparameters)
    coae.compile(optimizer_e=optimizer_e, optimizer_d=optimizer_d)
    return coae



def create_ae_model(hyperparameters=None):
    """

    Args:

    Returns:

    Raises:

    Examples:
    """
    if not hyperparameters:
        hyperparameters = keras_tuner.HyperParameters()

    latent_dim = 10
    shape = (70, 48, 24)

    activation = hyperparameters.Choice('activation',
        ['elu', 'tanh', 'silu'])
    num_encoder = hyperparameters.Int('num_encoder_layers',
                                      min_value=1, max_value=10, step=1,
                                      default=5)
    num_decoder = hyperparameters.Int('num_decoder_layers',
                                      min_value=1, max_value=10, step=1,
                                      default=5)
    batch_normalization = hyperparameters.Boolean('batch_normalization',
                                                  default=True)

    # Flatten dimensions
    flattened_dim = np.prod(shape)

    # Input layer is regular dimensions
    input = keras.Input(shape=shape, name='omni_flux_output')

    # Encoder layers
    encoded = layers.Flatten()(input)
    # Number of hiden layers
    for l in range(num_encoder):
        encoded = layers.Dense(hyperparameters.Int(
            'num_e_neurons_'+str(l), min_value=32, max_value=1_000, step=1,
            default=1_000),
                               activation=activation)(encoded)
        if batch_normalization:
            encoded = layers.BatchNormalization()(encoded)
    
    encoded = layers.Dense(latent_dim, activation='tanh',
                           name='latent_space')(encoded)

    encoder = keras.Model(input, encoded, name='encoder')


    encoded_input = keras.Input(shape=(latent_dim,), name='latent_space')
    decoded = layers.Dense(hyperparameters.Int(
        'num_d_neurons_'+str(0), min_value=32, max_value=1_000, step=1,
        default=1_000),
                           activation=activation)(encoded_input)
    for l in range(1, num_decoder):
        if batch_normalization:
            decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dense(hyperparameters.Int(
            'num_d_neurons_'+str(l), min_value=32, max_value=1_000, step=1,
            default=1_000),
                               activation=activation)(decoded)
    if batch_normalization:
        decoded = layers.BatchNormalization()(decoded)

    decoded = layers.Dense(flattened_dim, activation='linear')(decoded)
    decoded = layers.Reshape(shape, name='reconstruction')(decoded)

    decoder = keras.Model(encoded_input, decoded, name='decoder')

    model = keras.Model(input, decoder(encoder(input)), name='autoencoder')

    return model




@keras.saving.register_keras_serializable()
class CustomCMOrthoMetric(keras.Metric):

    def __init__(self, name='cm_i', factor_mape=1., **kwargs):
        super().__init__(name=name, **kwargs)
        self.cm = self.add_variable(
            shape=(),
            initializer='zeros',
            name='cm'
        )
        self.factor_mape = factor_mape

    def update_state(self, mse, latent, sample_weight=None):
        corr = tfp.stats.correlation(latent, latent)
        cm_term = tf.norm(corr - tf.eye(corr.shape[0]))
        mse_term = self.factor_mape*mse
        self.cm.assign(mse_term + cm_term)

    def result(self):
        return self.cm


@keras.saving.register_keras_serializable()
class OrthogonalAutoEncoder(keras.Model):
    """Make orthogonal autoencoder using tensorflow"""

    def __init__(self, hyperparameters=None,
                 latent_dim=100, shape=(70, 48, 24)):
        super(OrthogonalAutoEncoder, self).__init__()

        # SETUP
        # =====

        # HYPERPARAMETERS
        # ---------------
        if hyperparameters is None:
            hyperparameters = keras_tuner.HyperParameters()
        alpha = hyperparameters.Choice('alpha', [0.01, 0.1, 1., 10.])
        activation = hyperparameters.Choice('activation',
            ['elu', 'tanh', 'silu'])
        # activation = 'linear'
        num_encoder = hyperparameters.Int('num_encoder_layers',
                                          min_value=1, max_value=10, step=1,
                                          default=10)
        num_decoder = hyperparameters.Int('num_decoder_layers',
                                          min_value=1, max_value=10, step=1,
                                          default=10)
        batch_normalization = hyperparameters.Boolean('batch_normalization',
                                                      default=True)


        # Flatten dimensions
        flattened_dim = np.prod(shape)

        # Input layer is regular dimensions
        input = keras.Input(shape=shape, name='omni_flux_output')

        # Encoder layers
        encoded = layers.Flatten()(input)
        # Number of hiden layers
        for l in range(num_encoder):
            encoded = layers.Dense(hyperparameters.Int(
                'num_e_neurons_'+str(l), min_value=32, max_value=512, step=1,
                default=300),
                                   activation=activation)(encoded)
            if batch_normalization:
                encoded = layers.BatchNormalization()(encoded)
            # if dropout:
            #     encoded = layers.Dropout(dropout_rate)(encoded)
        
        encoded = layers.Dense(latent_dim, activation='tanh',
                               name='latent_space')(encoded)
        # encoded = layers.Dense(latent_dim, activation='linear',
        #                        name='latent_space')(encoded)

        encoder = keras.Model(input, encoded, name='encoder')

        # Decoder layers
        encoded_input = keras.Input(shape=(latent_dim,), name='latent_space')
        decoded = layers.Dense(hyperparameters.Int(
            'num_d_neurons_'+str(0), min_value=32, max_value=512, step=1,
            default=300),
                               activation=activation)(encoded_input)
        for l in range(1, num_decoder):
            if batch_normalization:
                decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Dense(hyperparameters.Int(
                'num_d_neurons_'+str(l), min_value=32, max_value=512, step=1),
                                   activation=activation)(decoded)
        if batch_normalization:
            decoded = layers.BatchNormalization()(decoded)
        # if dropout:
        #     decoded = layers.Dropout(dropout_rate)(decoded)
        decoded = layers.Dense(flattened_dim, activation='linear')(decoded)
        decoded = layers.Reshape(shape, name='reconstruction')(decoded)

        decoder = keras.Model(encoded_input, decoded, name='decoder')

        # Autoencoder
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

        # Losses
        self.alpha = alpha
        self.mse = keras.losses.MeanSquaredError()
        self.mape = keras.losses.MeanAbsolutePercentageError()
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.ortho_tracker = keras.metrics.Mean(name='orthogonal_loss')
        self.val_loss_tracker = keras.metrics.Mean(name='val_loss')
        self.cm_tracker = CustomCMOrthoMetric(factor_mape=0.7)
        self.val_cm_tracker = CustomCMOrthoMetric(factor_mape=0.7)
        self.mape_tracker = keras.metrics.MeanAbsolutePercentageError(
            name='mape')


    def get_config(self):
        return {"alpha": self.alpha,
                "latent_dim": self.latent_dim,
                "encoder": self.encoder,
                "decoder": self.decoder,
                }


    @property
    def metrics(self):
        return [self.loss_tracker,
                self.val_loss_tracker,
                self.mse_tracker,
                self.ortho_tracker,
                self.cm_tracker,
                self.val_cm_tracker,
                self.mape_tracker,
                ]

    def compile(self, optimizer_e, optimizer_d, **kwargs):
        super(OrthogonalAutoEncoder, self).compile(**kwargs)
        self.encoder.compile(optimizer=optimizer_e, **kwargs)
        self.decoder.compile(optimizer=optimizer_d, **kwargs)
        self.optimizer_e = optimizer_e
        self.optimizer_d = optimizer_d

    def call(self, x):
        return self.decoder(self.encoder(x))

    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            latent = self.encoder(x, training=True)
            xhat = self.decoder(latent, training=True)
            mse_loss = self.mse(y, xhat)  # compute
            # compute loss2 (orthogonal term)
            ortho_loss = tf.math.square(
                tf.linalg.matmul(
                    tf.transpose(latent), latent) - tf.eye(self.latent_dim))
            loss = mse_loss + self.alpha*ortho_loss
        grads_e = tape.gradient(loss, self.encoder.trainable_weights)
        grads_d = tape.gradient(loss, self.decoder.trainable_weights)
        self.optimizer_e.apply_gradients(zip(grads_e, self.encoder.trainable_weights))
        self.optimizer_d.apply_gradients(zip(grads_d, self.decoder.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(mse_loss)
        self.mape_tracker.update_state(x, xhat)
        self.ortho_tracker.update_state(ortho_loss)
        self.cm_tracker.update_state(self.mape_tracker.result(), latent)
        return {'loss': self.loss_tracker.result(),
                'mse': self.mse_tracker.result(),
                'mape': self.mape_tracker.result(),
                'ortho': self.ortho_tracker.result(),
                'cm': self.cm_tracker.result(),
                }

    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            latent = self.encoder(x, training=False)
            xhat = self.decoder(latent, training=False)
            mse_loss = self.mse(y, xhat)  # compute
            # compute loss2 (orthogonal term)
            ortho_loss = tf.math.square(
                tf.linalg.matmul(
                    tf.transpose(latent), latent) - tf.eye(self.latent_dim))
            loss = mse_loss + self.alpha*ortho_loss
        self.val_loss_tracker.update_state(loss)
        self.val_cm_tracker.update_state(self.mape(x, xhat), latent)
        return {'loss': self.val_loss_tracker.result(),
                'cm': self.val_cm_tracker.result()}


@keras.saving.register_keras_serializable()
class COAE(keras.Model):
    """Make convolutional orthogonal autoencoder using tensorflow"""

    def __init__(self, hyperparameters=None,
                 latent_dim=10, shape=(70, 48, 24)):
        super(COAE, self).__init__()

        # SETUP
        # =====

        # HYPERPARAMETERS
        # ---------------
        if hyperparameters is None:
            hyperparameters = keras_tuner.HyperParameters()
        alpha = hyperparameters.Choice('alpha', [0.01, 0.1, 1., 10.])
        activation = hyperparameters.Choice('activation',
            ['elu', 'tanh', 'silu'])
        
        # Flatten dimensions
        flattened_dim = np.prod(shape)

        # Input layer is regular dimensions
        input = keras.Input(shape=shape, name='omni_flux_output')
        encoded = layers.Reshape(list(shape)+[1])(input)

        # Encoder layers
        encoded = layers.Conv3D(16, (3, 3, 3),
                                activation=activation, padding='same')(encoded)
        encoded = layers.MaxPooling3D((2, 2, 2), padding='same')(encoded)
        encoded = layers.Conv3D(16, (3, 3, 3),
                                activation=activation, padding='same')(encoded)
        encoded = layers.MaxPooling3D((2, 2, 2), padding='same')(encoded)
        encoded = layers.Flatten()(encoded)
        
        encoded = layers.Dense(latent_dim, activation='tanh',
                               name='latent_space')(encoded)
        # encoded = layers.Dense(latent_dim, activation='linear',
        #                        name='latent_space')(encoded)

        encoder = keras.Model(input, encoded, name='encoder')
        encoder.summary()

        # Decoder layers
        encoded_input = keras.Input(shape=(latent_dim,), name='latent_space')
        decoded = layers.Dense(18*12*6*16, activation=activation)(encoded_input)
        decoded = layers.Reshape((18, 12, 6, 16))(decoded)
        decoded = layers.Conv3DTranspose(16, (3, 3, 3), strides=2,
                                         activation=activation,
                                         padding='same')(decoded)
        decoded = layers.Conv3DTranspose(16, (3, 3, 3), strides=2,
                                         activation=activation,
                                         padding='same')(decoded)
        decoded = layers.Conv3D(1, (3, 3, 3), activation=activation,
                                padding='same')(decoded)
        decoded = layers.Cropping3D((1, 0, 0))(decoded)
        decoded = layers.Reshape(shape)(decoded)

        decoder = keras.Model(encoded_input, decoded, name='decoder')
        decoder.summary()

        # Autoencoder
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

        # Losses
        self.alpha = alpha
        self.mse = keras.losses.MeanSquaredError()
        self.mape = keras.losses.MeanAbsolutePercentageError()
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.ortho_tracker = keras.metrics.Mean(name='orthogonal_loss')
        self.val_loss_tracker = keras.metrics.Mean(name='val_loss')
        self.cm_tracker = CustomCMOrthoMetric(factor_mape=0.7)
        self.val_cm_tracker = CustomCMOrthoMetric(factor_mape=0.7)
        self.mape_tracker = keras.metrics.MeanAbsolutePercentageError(
            name='mape')


    def get_config(self):
        return {"alpha": self.alpha,
                "latent_dim": self.latent_dim,
                "encoder": self.encoder,
                "decoder": self.decoder,
                }


    @property
    def metrics(self):
        return [self.loss_tracker,
                self.val_loss_tracker,
                self.mse_tracker,
                self.ortho_tracker,
                self.cm_tracker,
                self.val_cm_tracker,
                self.mape_tracker,
                ]

    def compile(self, optimizer_e, optimizer_d, **kwargs):
        super(COAE, self).compile(**kwargs)
        self.encoder.compile(optimizer=optimizer_e, **kwargs)
        self.decoder.compile(optimizer=optimizer_d, **kwargs)
        self.optimizer_e = optimizer_e
        self.optimizer_d = optimizer_d

    def call(self, x):
        return self.decoder(self.encoder(x))

    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            latent = self.encoder(x, training=True)
            xhat = self.decoder(latent, training=True)
            mse_loss = self.mse(y, xhat)  # compute
            # compute loss2 (orthogonal term)
            ortho_loss = tf.math.square(
                tf.linalg.matmul(
                    tf.transpose(latent), latent) - tf.eye(self.latent_dim))
            loss = mse_loss + self.alpha*ortho_loss
        grads_e = tape.gradient(loss, self.encoder.trainable_weights)
        grads_d = tape.gradient(loss, self.decoder.trainable_weights)
        self.optimizer_e.apply_gradients(zip(grads_e, self.encoder.trainable_weights))
        self.optimizer_d.apply_gradients(zip(grads_d, self.decoder.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(mse_loss)
        self.mape_tracker.update_state(x, xhat)
        self.ortho_tracker.update_state(ortho_loss)
        self.cm_tracker.update_state(self.mape_tracker.result(), latent)
        return {'loss': self.loss_tracker.result(),
                'mse': self.mse_tracker.result(),
                'mape': self.mape_tracker.result(),
                'ortho': self.ortho_tracker.result(),
                'cm': self.cm_tracker.result(),
                }

    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            latent = self.encoder(x, training=False)
            xhat = self.decoder(latent, training=False)
            mse_loss = self.mse(y, xhat)  # compute
            # compute loss2 (orthogonal term)
            ortho_loss = tf.math.square(
                tf.linalg.matmul(
                    tf.transpose(latent), latent) - tf.eye(self.latent_dim))
            loss = mse_loss + self.alpha*ortho_loss
        self.val_loss_tracker.update_state(loss)
        self.val_cm_tracker.update_state(self.mape(x, xhat), latent)
        return {'loss': self.val_loss_tracker.result(),
                'cm': self.val_cm_tracker.result()}


class UnweightedAEDataset(keras.utils.PyDataset):

    def __init__(self, x_set, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.x, = x_set
        self.batch_size = batch_size

    def __len__(self):
        # Return number of batches.
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        # Return x, y for batch idx.
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]

        return batch_x, batch_x

class WeightedChoiceAEDataset(keras.utils.Sequence):

    def __init__(self, trng, batch_size,
                 weights_choice, weights_loss=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.x = trng
        self.batch_size = batch_size
        self.weights_loss = weights_loss
        self.weights = weights_choice
        self.prob = weights_choice/np.sum(weights_choice)
        self.rng = np.random.default_rng()

    def __len__(self):
        # Return number of batches.
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        idxs = self.rng.choice(len(self.x),
                               size=self.batch_size, p=self.prob)
        x = self.x[idxs]
        # x and y are same because of autoencoder
        if self.weights_loss is None:
            return x, x
        # Include loss weights if given
        return x, x, self.weights_loss[idxs]


def uniform_weights_from_hist(samples, bins):
    """Return weights that would produce a uniform distribution when sampled.
    """
    weights = np.zeros(samples.shape)
    freq, edges = np.histogram(samples, bins=bins)
    freq_max = np.max(freq)
    for i in range(len(edges)-1):
        weight = freq_max/freq[i]
        weights[np.logical_and(samples>=edges[i], samples<edges[i+1])] = weight
        if i == len(edges)-2:
            weights[
                np.logical_and(samples>=edges[i], samples<=edges[i+1])
            ] = weight
    return weights



def tune_oae(train: WeightedChoiceAEDataset, val, **kwargs):
    """

    Args:

    Returns:

    Raises:

    Examples:
    """
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=create_oae_model,
        objective=keras_tuner.Objective('val_cm', direction='min'),
        directory="models/",
        project_name="cimi_oae",
        # distribution_strategy=tf.distribute.MirroredStrategy(),
        **kwargs
    )
    print(tuner.search_space_summary())
    # tuner.search(train, validation_data=(val, val),
    tuner.search(train, train, validation_data=(val, val),
                 epochs=1000, shuffle=True, batch_size=1024,
                 callbacks=[
                     EarlyStopping(patience=100, monitor='val_loss'),
                     ReduceLROnPlateau(patience=50),
                 ])
    return tuner


# def tune_linear_autoencoder(train: WeightedChoiceAEDataset, val, **kwargs):
#     """

#     Args:

#     Returns:

#     Raises:

#     Examples:
#     """
#     tuner = keras_tuner.BayesianOptimization(
#         hypermodel=create_model_linear,
#         objective=keras_tuner.Objective('val_loss', direction='min'),
#         directory="models/",
#         project_name="ram_scb_lae",
#         # distribution_strategy=tf.distribute.MirroredStrategy(),
#         **kwargs
#     )
#     print(tuner.search_space_summary())
#     tuner.search(train, validation_data=(val, val),
#                  epochs=1000, shuffle=True, batch_size=1024,
#                  callbacks=[
#                      EarlyStopping(patience=100, monitor='val_loss'),
#                      ReduceLROnPlateau(patience=50),
#                  ])
#     return tuner


def make_lstm_model(hp=None):
    """

    Args:
        hp:
            keras hyperparameters.

    Returns:
        (keras.models.Model): LSTM model.

    Raises:

    Examples:
    """
    input_shape = (3, 15)
    latent_dim = 10
    # Default architecture for testing
    if hp is None:
        num_lstm_units = 64
        num_lstm_layers = 1
        num_dense_layers = 3
        num_dense_neurons = 64
        batch_norm = False
        dropout = False
        optimizer = 'adam'
        activation = 'silu'
    # Hyperparameters
    else:
        num_lstm_units = hp.Int('num_lstm_neurons', 32, 300, step=4)
        num_lstm_layers = hp.Choice('num_lstm_layers', [1, 2], default=2)
        num_dense_layers = hp.Int('num_dense_layers', 1, 3, step=1, default=3)
        activation = hp.Choice('dense_activation', ['relu', 'elu', 'silu',
                                                    'sigmoid', 'softsign',
                                                    'softsign', 'softplus'])
        optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])

    model = Sequential(name='rope_model')
    model.add(layers.Input(shape=input_shape))

    if num_lstm_layers == 2:
        model.add(layers.LSTM(num_lstm_units, return_sequences=True))
        model.add(layers.LSTM(num_lstm_units))
    else:
        model.add(layers.LSTM(num_lstm_units))

    # Default is 3 layers 64 neurons each
    if hp is None:
        for i in range(num_dense_layers):
            model.add(layers.Dense(num_dense_neurons, activation=activation))
    # With hyperparameters
    else:
        for i in range(num_dense_layers):
            model.add(layers.Dense(hp.Int('num_dense_neurons_'+str(i),
                                          64, 600, step=4)))

    # Output
    model.add(layers.Dense(latent_dim, activation='linear'))

    model.compile(optimizer, loss='mean_squared_error',
                  metrics=['r2_score', 'root_mean_squared_error'])
    return model


def make_lstm_resnet_model(hp=None):
    """

    Args:
        hp:
            keras hyperparameters.

    Returns:
        (keras.models.Model): LSTM model.

    Raises:

    Examples:
    """
    input_shape = (3, 15)
    latent_dim = 10
    # Default architecture for testing
    if hp is None:
        hp = keras_tuner.HyperParameters()
        # num_lstm_units = 64
        # num_lstm_layers = 1
        # num_dense_layers = 3
        # num_dense_neurons = 64
        # batch_norm = False
        # dropout = False
        # optimizer = 'adam'
        # activation = 'silu'
    # Hyperparameters
    num_lstm_units = hp.Int('num_lstm_neurons', 32, 300, step=4)
    num_lstm_layers = hp.Choice('num_lstm_layers', [1, 2], default=2)
    num_dense_layers = hp.Int('num_dense_layers', 1, 3, step=1, default=3)
    activation = hp.Choice('dense_activation', ['relu', 'elu', 'silu',
                                                'sigmoid', 'softsign',
                                                'softsign', 'softplus'])
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])

    input = layers.Input(shape=input_shape)

    if num_lstm_layers == 2:
        model = layers.LSTM(num_lstm_units, return_sequences=True)(input)
        model = layers.LSTM(num_lstm_units)(model)
    else:
        model = layers.LSTM(num_lstm_units)(input)

    # Default is 3 layers 64 neurons each
    for i in range(num_dense_layers):
        model = layers.Dense(hp.Int('num_dense_neurons_'+str(i),
                                    64, 600, step=4))(model)

    # Add residual
    residual = layers.Dense(latent_dim, activation='linear')(model)
    x = layers.Reshape((3, 15, 1))(input)
    x = layers.Cropping2D(cropping=((2, 0), (0, 5)))(x)
    x = layers.Flatten()(x)
    output = layers.Add()([x, residual])
    # output = residual

    ret_model = keras.models.Model(inputs=input, outputs=output)

    ret_model.compile(optimizer, loss='mean_squared_error',
                      metrics=['r2_score', 'root_mean_squared_error'])
    return ret_model


def tune_lstm(train: tf.data.Dataset, val, **kwargs):
    """Tuner for the LSTM part of the model

    Args:

    Returns:

    Raises:

    Examples:
    """
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=make_lstm_model,
        objective="val_loss",
        directory="models/",
        project_name="ram_scb_rope",
        # distribution_strategy=tf.distribute.MirroredStrategy(),
        **kwargs
    )
    print(tuner.search_space_summary())
    tuner.search(train, validation_data=val,
                 epochs=5000,
                 callbacks=[
                     EarlyStopping(patience=50, monitor='val_loss'),
                     ReduceLROnPlateau(patience=10),
                 ])
    return tuner


def tune_lstm_resnet(train: tf.data.Dataset, val, **kwargs):
    """Tuner for the LSTM part of the model

    Args:

    Returns:

    Raises:

    Examples:
    """
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=make_lstm_resnet_model,
        objective="val_loss",
        directory="models/",
        project_name="ram_scb_resnet_rope",
        # distribution_strategy=tf.distribute.MirroredStrategy(),
        **kwargs
    )
    print(tuner.search_space_summary())
    tuner.search(train, validation_data=val,
                 epochs=5000,
                 callbacks=[
                     EarlyStopping(patience=500, monitor='val_loss'),
                     ReduceLROnPlateau(patience=100),
                 ])
    return tuner



def scale(data):
    """
    """
    return (data-1.5011928)/7.048491


def unscale(data):
    """
    """
    return data*7.048491+1.5011928


def predict_decoder(model, y):
    """
    """
    return unscale(model.layers[-1].predict(y))


def predict_encoder(model, y):
    """
    """
    return model.layers[1].predict(scale(y))


def predict_ae(model, y):
    """
    """
    return unscale(model.predict(scale(y)))


def to_sequences(x, sequence_length=1):
    "Return time series like array"
    ret_val = []
    for i in range(len(x) - sequence_length):
        ret_val.append(x[i:i+sequence_length])
    return np.array(ret_val)


def dynamic_prediction(model, data, forecast=6, lookback=3):
    """Create an hourly (default) dynamic prediction of latent space.
    """

    # Intialize 
    num_drivers = 5
    latent_dim = 10
    pred_dyn = []
    data_series = to_sequences(data, sequence_length=lookback)

    # Here be dragons
    t = 0
    while t < len(data_series)-forecast:
        # New latent spaces
        update = []
        current = deepcopy(
            data_series[t].reshape(1, 3, latent_dim+num_drivers))
        for f in range(forecast):
            if f == 1:
                # update latent space
                current[0, 2, :latent_dim] = update[-1]
                # update drivers
                current[0, 2, latent_dim:] = data_series[t][2, latent_dim:]
            elif f == 2:
                current[0, 2, :latent_dim] = update[-1]
                current[0, 1, :latent_dim] = update[-2]
                current[0, 2, latent_dim:] = data_series[t][2, latent_dim:]
                current[0, 1, latent_dim:] = data_series[t][1, latent_dim:]
            elif f > 2:
                current[0, 2, :latent_dim] = update[-1]
                current[0, 1, :latent_dim] = update[-2]
                current[0, 0, :latent_dim] = update[-3]
                current[0, 2, latent_dim:] = data_series[t][2, latent_dim:]
                current[0, 1, latent_dim:] = data_series[t][1, latent_dim:]
                current[0, 0, latent_dim:] = data_series[t][0, latent_dim:]

            pred = model.predict(current)
            update.append(pred)
            t = t+1

            pred_dyn.append(update[-1])

    return np.concatenate(pred_dyn)


def mdsa(truth, pred):
    """Calculate median symmetric accuracy
    
    Note:
        Make sure to take power of base 10 of log fluxes.
    """
    q = pred/truth
    ret_val = 100*((np.exp(np.median(abs(np.log(q)))))-1)
    return ret_val


def sspb(truth, pred):
    """Return Signed Symmetric Percentage Bias. 

    Note:
        Make sure to take power of base 10 of log fluxes.
    """
    q = pred/truth
    m_log_q = np.median(np.log(q))
    return 100*np.sign(m_log_q)*(np.exp(np.abs(m_log_q))-1)


def ensemble_pred(lstm_input, forecast=6):
    dynamic_prediction = [[0]*5]*5
    for i in range(5):
        for j in range(5):
            model = keras.models.load_model('models/rope_'+str(i)+'_'+str(j)+'.keras')
            pred_dyn = dynamic_prediction_piyush(model, lstm_input,
                                                 forecast=forecast)
            np.save('data/interim/pred_dyn_test_f'+str(forecast)+'_'+str(i)+'_'+str(j)+'.npy', pred_dyn)
            dynamic_prediction[i][j] = pred_dyn
    return np.array(dynamic_prediction)


def ensemble_mean_variance(dynamic_prediction, truth, weights):
    """

    Args:

    Returns:

    Raises:

    Examples:
    """
    w = weights
    z_ikt = np.zeros(dynamic_prediction.shape[1:])
    sigma2_ikt = np.zeros(dynamic_prediction.shape[1:])
    sigma2s_ikt = np.zeros(dynamic_prediction.shape[1:])
    s2_ik = np.zeros(dynamic_prediction.shape[2:])
    for i in range(z_ikt.shape[0]):
        z_ikt[i] = np.sum([w[i, j]*dynamic_prediction[i, j] for j in range(5)], axis=0)
        sigma2_ikt[i] = np.sum([w[i, j]*(z_ikt[i]-dynamic_prediction[i, j])**2 for j in range(5)], axis=0)
        s2_ik[i] = np.median(((truth-z_ikt[i])**2)/sigma2_ikt[i])
        sigma2s_ikt[i] = s2_ik[i]*sigma2_ikt[i]

    z_kt = np.mean(z_ikt, axis=0)
    sigma2_kt = np.mean(sigma2s_ikt, axis=0)

    return z_kt, sigma2_kt


def main(*args):
    """

    Args:

    Returns:

    Raises:

    Examples:
    """
    pass



if __name__ == '__main__':
    trng = np.load('data/trng.npy').astype('float32')
    val = np.load('data/val.npy').astype('float32')
    test = np.load('data/test.npy').astype('float32')
    trng_scaled = scale(trng)
    val_scaled = scale(val)
    test_scaled = scale(test)
    # dst = np.load('data/processed/symh_trng.npy')
    # bins = np.linspace(0, np.sqrt(250), 5)**2
    # weights = uniform_weights_from_hist(abs(dst), bins)
    # dataset = WeightedChoiceAEDataset(trng_scaled, 1024, weights)

    # main(sys.argv[1:])
