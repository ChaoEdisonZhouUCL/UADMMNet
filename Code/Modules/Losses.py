"""
Description:
    this is the module provides losses func.

"""

# Futures
from __future__ import print_function
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

import sys
import os

# getting the directory where this file is located
current_dir = os.path.dirname(os.path.realpath(__file__))
# getting the parent directory and adding it to the path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


__author__ = '{Chao ZHOU}'
__copyright__ = 'Copyright {04/09/2020}, {UADMMAENet}'
__email__ = '{chaozhouucl@gmail.com}'
__status__ = '{finished}'


# {code}
#########################################################################################################
# HSI unmixing losses
#########################################################################################################
def customize_loss(loss_weight):
    mse_lossweight = loss_weight[0]
    aad_lossweight = loss_weight[1]
    aid_lossweight = loss_weight[2]

    def calc_loss(y_true, y_pred):
        # MSE loss
        mse_loss = K.mean(K.square(y_true - y_pred))

        # AAD loss
        true = K.sqrt(K.sum(K.square(y_true), axis=-1))
        pred = K.sqrt(K.sum(K.square(y_pred), axis=-1))

        denominator = tf.multiply(true, pred) + 1e-8
        numerator = K.sum(tf.multiply(y_pred, y_true), axis=-1) + 1e-8
        frac = tf.divide(numerator, denominator)

        frac = K.maximum(frac, -0.99)
        frac = K.minimum(frac, .99)
        aad_loss = tf.acos(frac) * 180.0 / 3.14

        # AID loss
        divergence_1 = tf.divide(y_true, y_pred)
        divergence_1 = K.sum(tf.multiply(K.log(divergence_1), y_true), axis=-1)

        divergence_2 = tf.divide(y_pred, y_true)
        divergence_2 = K.sum(tf.multiply(K.log(divergence_2), y_pred), axis=-1)

        aid_loss = divergence_1 + divergence_2

        # sum all loss
        loss = mse_lossweight * mse_loss + aad_lossweight * \
            aad_loss + aid_lossweight * aid_loss
        return loss

    return calc_loss

#########################################################################################################
# HSI unmixing metrics
#########################################################################################################


def RMSE_metric(y_true, y_pred):
    MSE = np.mean(np.square(y_true - y_pred), axis=-1)
    rmse = np.sqrt(MSE)
    return np.mean(rmse)


def MAE_metric(y_true, y_pred):
    MAE = np.abs(y_true - y_pred)*100
    return np.mean(MAE)


def angle_distance_metric(y_true, y_pred, verbose=False):
    '''

    :param y_true: (No_Endm, No_Bands)
    :param y_pred: (No_Endm, No_Bands)
    :return:
    '''
    dot_product = np.sum(y_true * y_pred, axis=-1)
    l2_norms = np.linalg.norm(y_true, axis=-1) * \
        np.linalg.norm(y_pred, axis=-1) + 1e-8
    cosine_similarity = dot_product / l2_norms

    # Clamp the cosine similarity to the range [-1, 1] to avoid NaN values
    eps = 1e-7
    cosine_similarity = np.clip(
        cosine_similarity, a_min=-1.0 + eps, a_max=1.0 - eps)
    AAD = np.arccos(cosine_similarity) * 180.0 / np.pi

    if verbose:
        print(f"angle distance is: {AAD}")
        return AAD, np.mean(AAD)
    else:
        return np.mean(AAD)


def abundance_information_divergence_metric(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-8)
    divergence_1 = np.divide(y_true, y_pred)
    divergence_1 = np.sum(np.multiply(np.log(divergence_1), y_true), axis=-1)

    divergence_2 = np.divide(y_pred, y_true)

    divergence_2 = np.sum(np.multiply(np.log(divergence_2), y_pred), axis=-1)

    AID = divergence_1 + divergence_2

    return np.mean(AID)
