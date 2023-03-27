"""
Description:
this file provides network model.

"""

# Futures
from __future__ import print_function

import sys
import os

# getting the directory where this file is located
current_dir = os.path.dirname(os.path.realpath(__file__))
# getting the parent directory and adding it to the path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from Modules.utils import *
from Modules.Losses import *
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense
from tensorflow.keras.constraints import Constraint
import tensorflow.keras.initializers as initializers
import tensorflow.keras.backend as K
import tensorflow as tf
import scipy.linalg as splin
import scipy as sp
import numpy as np



__author__ = '{Chao ZHOU}'
__copyright__ = 'Copyright {04/09/2020}, {UADMMAENet}'
__email__ = '{chaozhouucl@gmail.com}'
__status__ = '{finished}'


# {code}
# ---------------------------------------------------------------------------------------------
# customize keras layer for unfolding ADMM
# ---------------------------------------------------------------------------------------------
class UADMMNet_Reconstruction_layer(Layer):
    def __init__(self, units, W_initializer, B_initializer, **kwargs):
        self.units = units
        self.W_initializer = initializers.get(W_initializer)
        self.B_initializer = initializers.get(B_initializer)

        super(UADMMNet_Reconstruction_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # parse shape
        y_shape, _, _ = input_shape
        obs_dim = y_shape[-1]

        # add trainable params
        self.W = self.add_weight(name='W',
                                 shape=(obs_dim, self.units),
                                 initializer=self.W_initializer,
                                 trainable=True)

        self.B = self.add_weight(name='B',
                                 shape=(self.units, self.units),
                                 initializer=self.B_initializer,
                                 trainable=True)
        self.built = True

    def call(self, x):
        assert isinstance(x, list)
        y, z, d = x

        output = K.dot(y, self.W) + K.dot(z + d, self.B)
        # check the output_dim = z_dim
        assert (K.int_shape(z) == K.int_shape(output))
        return output

    def get_config(self):
        config = {
            'units': self.units,
            'W_initializer': initializers.serialize(self.W_initializer),
            'B_initializer': initializers.serialize(self.B_initializer),
        }
        base_config = super(UADMMNet_Reconstruction_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # parse shape
        _, z_shape, _ = input_shape
        return z_shape


class UADMMNet_AuxiliaryVariableUpdate_layer(Layer):
    def __init__(self, units, scalar_threshold, threshold_initializer, **kwargs):
        self.units = units
        self.scalar_threshold = scalar_threshold
        self.threshold_initializer = initializers.get(threshold_initializer)

        super(UADMMNet_AuxiliaryVariableUpdate_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # parse input_shape
        x_shape, _ = input_shape
        code_dim = x_shape[-1]
        assert code_dim == self.units
        # define the shape of threshold
        if self.scalar_threshold:  # scalar threshold
            self.threshold_shape = (1,)
        else:
            self.threshold_shape = (code_dim,)

        # add trainable params
        self.threshold = self.add_weight(name='threshold',
                                         shape=self.threshold_shape,
                                         initializer=self.threshold_initializer,
                                         trainable=True)

        self.built = True  # Be sure to call this at the end

    def call(self, x):
        assert (isinstance(x, list))
        input, d = x
        # threshold = K.abs(self.threshold)
        return K.relu(input - d - self.threshold)

    def get_config(self):
        config = {
            'units': self.units,
            'scalar_threshold': self.scalar_threshold,
            'threshold_initializer': initializers.serialize(self.threshold_initializer)
        }
        base_config = super(
            UADMMNet_AuxiliaryVariableUpdate_layer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):

        return input_shape[0]


class UADMMNet_Multiplier_layer(Layer):
    def __init__(self, units, scalar_eta, eta_initializer, **kwargs):
        self.units = units
        self.scalar_eta = scalar_eta
        self.eta_initializer = initializers.get(eta_initializer)

        super(UADMMNet_Multiplier_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        x_shape, _, _ = input_shape
        code_dim = x_shape[-1]
        assert code_dim == self.units
        # define the shape of threshold
        if self.scalar_eta:  # scalar threshold
            self.eta_shape = (1,)
        else:
            self.eta_shape = (code_dim,)
        # Create a trainable weight variable for this layer.
        self.eta = self.add_weight(name='eta',
                                   shape=self.eta_shape,
                                   initializer=self.eta_initializer,
                                   trainable=True)

        self.built = True  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        input, z, d = x
        return d - (input - z) * self.eta

    def get_config(self):
        config = {
            'units': self.units,
            'scalar_eta': self.scalar_eta,
            'eta_initializer': initializers.serialize(self.eta_initializer)
        }
        base_config = super(UADMMNet_Multiplier_layer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        _, _, d_input_shape = input_shape
        return d_input_shape


class UADMMNet_Norm_layer(Layer):
    def __init__(self, **kwargs):
        super(UADMMNet_Norm_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True  # Be sure to call this at the end

    def call(self, x):
        x = x + 1e-8
        return x / tf.norm(x, ord=1, axis=-1, keepdims=True)

    def get_config(self):
        base_config = super(UADMMNet_Norm_layer, self).get_config()

        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape


# ---------------------------------------------------------------------------------------------
# valina unfolding ADMM network for abundance estimation, fully supervised
# ---------------------------------------------------------------------------------------------
class UADMM_AENet:
    def __init__(self, input_shape, output_shape, number_layers,  share_layers,
                 name='UADMM_AENet', A=None, lambda_0=None, scalar_threshold=True, scalar_eta=True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.number_layers = number_layers
        self.scalar_threshold = scalar_threshold
        self.scalar_eta = scalar_eta
        self.share_layers = share_layers
        self.name = name
        self.A = A
        self.lambda_0 = lambda_0

        # initialize with prior information: (1)endm: A; and (2) ADMM params: lambda.
        if self.A is not None and self.lambda_0 is not None:
            obs_dim, code_dim = self.A.shape
            assert code_dim == self.output_shape[0]
            norm_m = splin.norm(self.A) * (25 + code_dim) / float(code_dim)
            # rescale endm matrix and lambda
            self.A = self.A / norm_m
            self.lambda_0 = self.lambda_0 / norm_m ** 2

            # calc the correpsonding mu
            mu_AL = 0.01
            mu = 10 * self.lambda_0 + mu_AL

            # initial value for threshold, which is the learnable params in Auxiliary variable update layer
            threshold = self.lambda_0 / mu
            if self.scalar_threshold:
                self.threshold_initializer = initializers.constant(threshold)
            else:
                self.threshold_initializer = initializers.constant(
                    threshold * np.ones(code_dim, ))

            # initial value for W and B, which is the learnable params in Reconstruction layer
            [UF, SF] = splin.svd(sp.dot(self.A.T, self.A))[:2]
            IF = sp.dot(sp.dot(UF, sp.diag(1. / (SF + mu))), UF.T)
            W = sp.dot(self.A, IF) / norm_m
            B = IF * mu
            self.W_initializer = initializers.constant(W)
            self.B_initializer = initializers.constant(B)

            # initial value for eta, which is the learnable params in dual variale update layer
            if self.scalar_eta:
                self.eta_initializer = initializers.constant(np.ones(1, ))
            else:
                self.eta_initializer = initializers.constant(
                    np.ones(code_dim, ))

        else:  # random initialization
            # initial value for threshold, which is the learnable params in Auxiliary variable update layer
            self.threshold_initializer = initializers.glorot_normal()
            # initial value for W and B, which is the learnable params in Reconstruction layer
            self.W_initializer = initializers.glorot_normal()
            self.B_initializer = initializers.glorot_normal()
            # initial value for eta, which is the learnable params in dual variale update layer
            self.eta_initializer = initializers.glorot_normal()

        self.build()

    def build(self):

        # start building UADMM-AENet
        input = Input(shape=self.input_shape, name=self.name + '_Input_layer')
        z_0 = Input(shape=self.output_shape, name=self.name + '_z0Input_layer')
        d_0 = Input(shape=self.output_shape, name=self.name + '_d0Input_layer')

        # build shared layer if true
        if self.share_layers:
            shared_Recon_layer = UADMMNet_Reconstruction_layer(self.output_shape[0], self.W_initializer,
                                                               self.B_initializer,
                                                               name=self.name + 'Shared_Recon_Layer')
            shared_AuxUpdate_layer = UADMMNet_AuxiliaryVariableUpdate_layer(self.output_shape[0],
                                                                            self.scalar_threshold,
                                                                            self.threshold_initializer,
                                                                            name=self.name + 'AuxVarUpdate_Layer')
            shared_MultiplierUpdate_layer = UADMMNet_Multiplier_layer(self.output_shape[0], self.scalar_eta,
                                                                      self.eta_initializer,
                                                                      name=self.name + 'MultiplierUpdate_Layer')

            x_k = shared_Recon_layer([input, z_0, d_0])
            z_k = shared_AuxUpdate_layer([x_k, d_0])
            d_k = shared_MultiplierUpdate_layer([x_k, z_k, d_0])

            for i in range(self.number_layers - 1):
                x_k = shared_Recon_layer([input, z_k, d_k])
                z_k = shared_AuxUpdate_layer([x_k, d_k])
                d_k = shared_MultiplierUpdate_layer([x_k, z_k, d_k])

            x_k = shared_Recon_layer([input, z_k, d_k])
            output = shared_AuxUpdate_layer([x_k, d_k])
        else:
            x_k = UADMMNet_Reconstruction_layer(self.output_shape[0], self.W_initializer,
                                                self.B_initializer,
                                                name=self.name + 'Recon_Layer_1')([input, z_0, d_0])
            z_k = UADMMNet_AuxiliaryVariableUpdate_layer(self.output_shape[0],
                                                         self.scalar_threshold,
                                                         self.threshold_initializer,
                                                         name=self.name + 'AuxVarUpdate_Layer_1')([x_k, d_0])
            d_k = UADMMNet_Multiplier_layer(self.output_shape[0], self.scalar_eta,
                                            self.eta_initializer,
                                            name=self.name + 'MultiplierUpdate_Layer_1')([x_k, z_k, d_0])

            for i in range(self.number_layers - 1):
                x_k = UADMMNet_Reconstruction_layer(self.output_shape[0], self.W_initializer,
                                                    self.B_initializer,
                                                    name=self.name + 'Recon_Layer_{}'.format(i + 2))(
                    [input, z_k, d_k])
                z_k = UADMMNet_AuxiliaryVariableUpdate_layer(self.output_shape[0],
                                                             self.scalar_threshold,
                                                             self.threshold_initializer,
                                                             name=self.name + 'AuxVarUpdate_Layer_{}'.format(i + 2))(
                    [x_k, d_k])
                d_k = UADMMNet_Multiplier_layer(self.output_shape[0], self.scalar_eta,
                                                self.eta_initializer,
                                                name=self.name + 'MultiplierUpdate_Layer_{}'.format(i + 2))(
                    [x_k, z_k, d_k])
            x_k = UADMMNet_Reconstruction_layer(self.output_shape[0], self.W_initializer,
                                                self.B_initializer,
                                                name=self.name + 'Output_Recon_Layer')(
                [input, z_k, d_k])
            output = UADMMNet_AuxiliaryVariableUpdate_layer(self.output_shape[0],
                                                            self.scalar_threshold,
                                                            self.threshold_initializer,
                                                            name=self.name + 'Output_AuxVarUpdate_Layer')([x_k, d_k])
        output = UADMMNet_Norm_layer(
            name=self.name + 'Norm_Output_Layer')(output)

        self.model = Model(inputs=[input, z_0, d_0],
                           outputs=[output], name=self.name)


# ---------------------------------------------------------------------------------------------
# customize endm signature constraint： constrain endm signature in range [0,1]
# ---------------------------------------------------------------------------------------------
class MinMaxVal(Constraint):
    """Constrains the weights to be no less than min_value.
        """

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        temp = w

        temp = K.maximum(temp, self.min_value)

        temp = K.minimum(temp, self.max_value)
        return temp

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value,

                }


# ---------------------------------------------------------------------------------------------
# valina unfolding ADMM network for blind unmixing, fully unsupervised
# ---------------------------------------------------------------------------------------------
class UADMM_BUNet:
    def __init__(self, input_shape, output_shape, number_layers, share_layers,
                 name='UADMM_BUNet', Endm_initialization=None, A=None, lambda_0=None, scalar_threshold=True, scalar_eta=True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.number_layers = number_layers
        self.scalar_threshold = scalar_threshold
        self.scalar_eta = scalar_eta
        self.share_layers = share_layers
        self.name = name
        self.Endm_initialization = Endm_initialization
        self.A = A
        self.lambda_0 = lambda_0

        # initial value for endm layer
        if Endm_initialization is not None:
            self.Endm_kernel_initializer = initializers.constant(
                self.Endm_initialization.T)
        else:
            self.Endm_kernel_initializer = initializers.glorot_uniform()

        # initialize with prior information: (1)endm: A; and (2) ADMM params: lambda.
        if self.A is not None and self.lambda_0 is not None:
            obs_dim, code_dim = self.A.shape
            assert code_dim == self.output_shape[0]
            norm_m = splin.norm(self.A) * (25 + code_dim) / float(code_dim)
            # rescale endm matrix and lambda
            self.A = self.A / norm_m
            self.lambda_0 = self.lambda_0 / norm_m ** 2

            # calc the correpsonding mu
            mu_AL = 0.01
            mu = 10 * self.lambda_0 + mu_AL

            # initial value for threshold, which is the learnable params in Auxiliary variable update layer
            threshold = self.lambda_0 / mu
            if self.scalar_threshold:
                self.threshold_initializer = initializers.constant(threshold)
            else:
                self.threshold_initializer = initializers.constant(
                    threshold * np.ones(code_dim, ))

            # initial value for W and B, which is the learnable params in Reconstruction layer
            [UF, SF] = splin.svd(sp.dot(self.A.T, self.A))[:2]
            IF = sp.dot(sp.dot(UF, sp.diag(1. / (SF + mu))), UF.T)
            W = sp.dot(self.A, IF) / norm_m
            B = IF * mu
            self.W_initializer = initializers.constant(W)
            self.B_initializer = initializers.constant(B)

            # initial value for eta, which is the learnable params in dual variale update layer
            if self.scalar_eta:
                self.eta_initializer = initializers.constant(np.ones(1, ))
            else:
                self.eta_initializer = initializers.constant(
                    np.ones(code_dim, ))

        else:  # random initialization
            # initial value for endm layer
            self.Endm_kernel_initializer = initializers.glorot_uniform()
            # initial value for threshold, which is the learnable params in Auxiliary variable update layer
            self.threshold_initializer = initializers.glorot_normal()
            # initial value for W and B, which is the learnable params in Reconstruction layer
            self.W_initializer = initializers.glorot_normal()
            self.B_initializer = initializers.glorot_normal()
            # initial value for eta, which is the learnable params in dual variale update layer
            self.eta_initializer = initializers.glorot_normal()

        self.build()

    def build(self):

        # start building UADMM-BUNet
        input = Input(shape=self.input_shape, name=self.name + '_Input_layer')
        z_0 = Input(shape=self.output_shape, name=self.name + '_z0Input_layer')
        d_0 = Input(shape=self.output_shape, name=self.name + '_d0Input_layer')

        # build shared layer if true
        if self.share_layers:
            shared_Recon_layer = UADMMNet_Reconstruction_layer(self.output_shape[0], self.W_initializer,
                                                               self.B_initializer,
                                                               name=self.name + 'Shared_Recon_Layer')
            shared_AuxUpdate_layer = UADMMNet_AuxiliaryVariableUpdate_layer(self.output_shape[0],
                                                                            self.scalar_threshold,
                                                                            self.threshold_initializer,
                                                                            name=self.name + 'AuxVarUpdate_Layer')
            shared_MultiplierUpdate_layer = UADMMNet_Multiplier_layer(self.output_shape[0], self.scalar_eta,
                                                                      self.eta_initializer,
                                                                      name=self.name + 'MultiplierUpdate_Layer')

            x_k = shared_Recon_layer([input, z_0, d_0])
            z_k = shared_AuxUpdate_layer([x_k, d_0])
            d_k = shared_MultiplierUpdate_layer([x_k, z_k, d_0])

            for i in range(self.number_layers - 1):
                x_k = shared_Recon_layer([input, z_k, d_k])
                z_k = shared_AuxUpdate_layer([x_k, d_k])
                d_k = shared_MultiplierUpdate_layer([x_k, z_k, d_k])

            x_k = shared_Recon_layer([input, z_k, d_k])
            output = shared_AuxUpdate_layer([x_k, d_k])
        else:
            x_k = UADMMNet_Reconstruction_layer(self.output_shape[0], self.W_initializer,
                                                self.B_initializer,
                                                name=self.name + 'Recon_Layer_1')([input, z_0, d_0])
            z_k = UADMMNet_AuxiliaryVariableUpdate_layer(self.output_shape[0],
                                                         self.scalar_threshold,
                                                         self.threshold_initializer,
                                                         name=self.name + 'AuxVarUpdate_Layer_1')([x_k, d_0])
            d_k = UADMMNet_Multiplier_layer(self.output_shape[0], self.scalar_eta,
                                            self.eta_initializer,
                                            name=self.name + 'MultiplierUpdate_Layer_1')([x_k, z_k, d_0])

            for i in range(self.number_layers - 1):
                x_k = UADMMNet_Reconstruction_layer(self.output_shape[0], self.W_initializer,
                                                    self.B_initializer,
                                                    name=self.name + 'Recon_Layer_{}'.format(i + 2))(
                    [input, z_k, d_k])
                z_k = UADMMNet_AuxiliaryVariableUpdate_layer(self.output_shape[0],
                                                             self.scalar_threshold,
                                                             self.threshold_initializer,
                                                             name=self.name + 'AuxVarUpdate_Layer_{}'.format(i + 2))(
                    [x_k, d_k])
                d_k = UADMMNet_Multiplier_layer(self.output_shape[0], self.scalar_eta,
                                                self.eta_initializer,
                                                name=self.name + 'MultiplierUpdate_Layer_{}'.format(i + 2))(
                    [x_k, z_k, d_k])
            x_k = UADMMNet_Reconstruction_layer(self.output_shape[0], self.W_initializer,
                                                self.B_initializer,
                                                name=self.name + 'Output_Recon_Layer')(
                [input, z_k, d_k])
            output = UADMMNet_AuxiliaryVariableUpdate_layer(self.output_shape[0],
                                                            self.scalar_threshold,
                                                            self.threshold_initializer,
                                                            name=self.name + 'Output_AuxVarUpdate_Layer')([x_k, d_k])
        output = UADMMNet_Norm_layer(
            name=self.name + 'Norm_Output_Layer')(output)
        output = Dense(units=self.input_shape[0],
                       use_bias=False,
                       kernel_initializer=self.Endm_kernel_initializer,
                       kernel_constraint=MinMaxVal(
                           min_value=1e-6, max_value=1.0),
                       name=self.name + 'Endm_layer')(output)
        self.model = Model(inputs=[input, z_0, d_0],
                           outputs=[output], name=self.name)


##############################################################################################################
# define callbacks to visualize the decoder weights/endm_sig and the abundance map.
##############################################################################################################
class Collect_intermediate_value(Callback):
    def __init__(self, log_dir, val_x, valid_epoch, **kwargs):
        self.log_dir = log_dir
        self.val_x = val_x
        self.valid_epoch = valid_epoch

        self.est_endm_sig = {}
        self.est_abundance = {}
        super(Collect_intermediate_value, self).__init__(*kwargs)

    def collect_est_endm_sig(self, title):
        est_endm_sig = self.decoder_layer.get_weights()[0].T
        self.est_endm_sig[title] = est_endm_sig

    def collect_est_abundance(self, title):
        est_abundance = self.Encoder.predict(self.val_x, batch_size=1024)
        self.est_abundance[title] = est_abundance

    def on_train_begin(self, logs=None):
        # get decoder layer where the weights
        # is the est endmember signature in
        # linear mixing model
        self.decoder_layer = self.model.layers[-1]

        # get the encoder model
        # that output the latent distribution parameter
        for layer in self.model.layers:
            if 'Norm_Output_Layer' in layer.name:
                encoder_output_layer = layer
                break
        self.Encoder = Model(self.model.input, encoder_output_layer.output)

        # collect estimated value
        self.collect_est_endm_sig(title='epoch0')
        self.collect_est_abundance(title='epoch0')
        super(Collect_intermediate_value, self).on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.collect_est_endm_sig(title='finalepoch')
        self.collect_est_abundance(title='finalepoch')
        write_data(data=self.est_endm_sig, file_path=self.log_dir +
                   'intermediate_est_Endm_sig.pkl')
        write_data(data=self.est_abundance, file_path=self.log_dir +
                   'intermediate_est_abundance.pkl')
        super(Collect_intermediate_value, self).on_train_end(logs)

    def on_epoch_end(self, epoch, logs=None):
        # collect estimated value
        if epoch % self.valid_epoch == 0:
            self.collect_est_endm_sig(title='epoch{}'.format(epoch))
            self.collect_est_abundance(title='epoch{}'.format(epoch))
            write_data(data=self.est_endm_sig,
                       file_path=self.log_dir + 'intermediate_est_Endm_sig.pkl')
            write_data(data=self.est_abundance,
                       file_path=self.log_dir + 'intermediate_est_abundance.pkl')
        super(Collect_intermediate_value, self).on_epoch_end(epoch, logs)


##############################################################################################################
# define callbacks to print and summary performance report into Readme.txt file
##############################################################################################################
class Print_intermediate_value(Callback):
    def __init__(self, readme_path, true_endm_sig, true_abundance, val_x, valid_epoch, **kwargs):
        self.readme_path = readme_path
        self.val_x = val_x
        self.true_abundance = true_abundance
        self.true_endm_sig = true_endm_sig
        self.valid_epoch = valid_epoch

        super(Print_intermediate_value, self).__init__(**kwargs)

    def collect_est_endm_sig(self):
        est_endm_sig = self.decoder_layer.get_weights()[0].T
        asq = Endmember_reorder2(self.true_endm_sig, est_endm_sig)
        est_endm_sig = est_endm_sig[:, asq]
        return est_endm_sig

    def collect_est_abundance(self):
        est_abundance = self.Encoder.predict(self.val_x, batch_size=64)
        return est_abundance

    def summary(self, epoch):
        # collect estimated endm
        est_endm_sig = self.collect_est_endm_sig()
        # calc metrics
        sad = angle_distance_metric(self.true_endm_sig.T, est_endm_sig.T)
        # print and summary SAD
        summary_str = 'Epoch ' + str(epoch) + \
            ' est Endms\r\n' + ' SAD: %f\r\n' % (sad)
        print(summary_str)
        summary2readme(
            summary_str, readme_path=self.readme_path + 'Readme.txt')

        # collect estimated abundance
        est_abundance = self.collect_est_abundance()
        # calc metricscollect estimated endm
        rmse = RMSE_metric(self.true_abundance, est_abundance)
        aad = angle_distance_metric(self.true_abundance, est_abundance)
        aid = abundance_information_divergence_metric(
            self.true_abundance, est_abundance)

        # print and summary RMSE
        summary_str = 'Epoch ' + str(epoch) + ' est Abundance\r\n' + ' RMSE: %f\r\n' % (rmse) \
                      + 'AAD: %f\r\n' % (aad) \
                      + 'AID: %f\r\n' % (aid)
        print(summary_str)

        summary2readme(
            summary_str, readme_path=self.readme_path + 'Readme.txt')

    def on_train_begin(self, logs=None):
        # get decoder layer where the weights
        # is the est endmember signature in
        # linear mixing model
        self.decoder_layer = self.model.layers[-1]

        # get the encoder model
        # that output the abundance
        self.Encoder = Model(self.model.input, self.model.layers[-2].output)

        super(Print_intermediate_value, self).on_train_begin(logs)

    def on_train_end(self, logs=None):
        super(Print_intermediate_value, self).on_train_end(logs)

    def on_epoch_end(self, epoch, logs=None):
        # collect estimated value
        if epoch % self.valid_epoch == 0:
            self.summary(epoch)
        super(Print_intermediate_value, self).on_epoch_end(epoch, logs)
