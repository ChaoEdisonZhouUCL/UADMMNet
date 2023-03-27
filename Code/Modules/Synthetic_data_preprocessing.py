"""
Description:
this file provides synthetic data pre-processing funcs.
"""

# Futures
from __future__ import print_function
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


########################################################################################################################
# multivariate Gaussian noise
########################################################################################################################

def add_gaussian_noise(signal, snr):
    # calc the power of noise (sigma^2) according to SNR and data
    NO_DATA, NO_Bands = signal.shape
    signal_power = np.mean(np.diagonal(np.dot(signal, signal.T)))
    noise_power = signal_power / np.power(10, snr / 10)
    # mean
    mean = np.zeros(shape=NO_Bands)
    # cov
    cov = np.eye(NO_Bands)
    # multivariate gaussian noise
    noise = np.sqrt(noise_power / NO_Bands) * \
        np.random.multivariate_normal(mean=mean, cov=cov, size=NO_DATA)

    noise_sig = signal + noise

    return noise_sig


########################################################################################################################
# sampling strategies
########################################################################################################################
def sampling(NO_DATA, sample_size, sampling_strategy='random'):
    '''

    :param NO_DATA:
    :param sample_size:
    :param sampling_strategy:
    1. fixed sampling: get the first (sample_size) pixel index from pixel_indexes={0,1,2,3,4,5...};

    2. random_sampling: random sample (sample_size) pixel index from pixel_indexes={0,1,2,3,4,5...};

    :return: list of sample pixel index.
    '''
    if sampling_strategy == 'fixed':

        index = np.arange(NO_DATA)
        train_index = index[:sample_size]
        eval_index = index[sample_size:]
        train_index = list(train_index)
        eval_index = list(eval_index)
    else:
        index = np.arange(NO_DATA)
        np.random.shuffle(index)
        train_index = index[:sample_size]
        eval_index = index[sample_size:]
        train_index = list(train_index)
        eval_index = list(eval_index)

    return train_index, eval_index
