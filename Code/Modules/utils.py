"""
Description:
    this is the module provides utils functions for unfolding SunSAL network.

"""

# Futures
from __future__ import print_function

import datetime
import os
import pickle
import sys

import numpy as np
import tensorflow as tf
from scipy.io import savemat


__author__ = '{Chao ZHOU}'
__copyright__ = 'Copyright {04/09/2020}, {UADMMAENet}'
__email__ = '{chaozhouucl@gmail.com}'
__status__ = '{finished}'

# {code}
##############################################################################################################
# define project log path creation function
##############################################################################################################
# create project log path


def create_project_log_path(project_path, **kwargs):
    # year_month_day/hour_min/(model_log_dir, model_checkpoint_dir, tensorboard_log_dir)/
    date = datetime.datetime.now()

    program_day = project_path + date.strftime("%Y_%m_%d")
    if not os.path.exists(program_day):
        os.mkdir(program_day)

    Readme_flag = False
    if 'Readme' in kwargs.keys():
        Readme_flag = True
    if Readme_flag:
        readme = kwargs.pop('Readme')

    program_time = date.strftime("%H_%M_%S")
    for key, value in kwargs.items():
        program_time = program_time + '_' + key + '_{}'.format(value)

    program_log_parent_dir = os.path.join(program_day, program_time + '/')
    if not os.path.exists(program_log_parent_dir):
        os.mkdir(program_log_parent_dir)

    # model checkpoint dir
    model_checkpoint_dir = program_log_parent_dir + 'model_checkpoint_dir/'
    if not os.path.exists(model_checkpoint_dir):
        os.mkdir(model_checkpoint_dir)

    # tensorboard_log_dir
    tensorboard_log_dir = program_log_parent_dir + 'tensorboard_log_dir/'
    if not os.path.exists(tensorboard_log_dir):
        os.mkdir(tensorboard_log_dir)

    # model_log_dir
    model_log_dir = program_log_parent_dir + 'model_log_dir/'
    if not os.path.exists(model_log_dir):
        os.mkdir(model_log_dir)

    # write exp log
    if Readme_flag:
        with open(program_log_parent_dir + 'Readme.txt', 'w') as f:
            f.write(readme + '\r\n')
            for key, value in kwargs.items():
                f.write(key + ': {}'.format(value))
            f.write('program log dir: ' + program_log_parent_dir + '\r\n')

    return program_log_parent_dir, model_checkpoint_dir, tensorboard_log_dir, model_log_dir


# write model summary to readme.txt
def summary_model2_readme(model, readme_path):
    with open(readme_path, 'a') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


# write summary to readme.txt
def summary2readme(summary, readme_path):
    with open(readme_path, 'a') as fh:
        fh.write(summary + '\r\n')


##############################################################################################################
# define functions that write/read data into/from pickle
##############################################################################################################
# define data writer func
def write_data(data, file_path):
    file_writer = open(file_path, 'wb')
    pickle.dump(data, file_writer)
    file_writer.close()


# define data reader func
def read_data(file_path):
    file_reader = open(file_path, 'rb')
    data = pickle.load(file_reader)
    file_reader.close()
    return data


##############################################################################################################
# define functions that convert between pkl, np, and mat
##############################################################################################################
# convert pkl file to mat file
def pkl2mat(filepath):
    data = read_data(filepath)
    for key, value in data.items():
        if type(value) != np.ndarray:
            data[key] = value.numpy()

    filepath = filepath.split('.')
    filepath = filepath[0].split('/')
    data_name = filepath[-1]
    filepath = '/'.join(filepath)
    filepath = filepath + '.mat'
    data = {data_name: data}
    savemat(filepath, data)


# onvert npy file to mat
def np2mat(filepath):
    data = np.load(filepath)
    filepath = filepath.split('.')
    filepath = filepath[0].split('/')
    data_name = filepath[-1]
    filepath = '/'.join(filepath)
    filepath = filepath + '.mat'
    data = {data_name: data}
    savemat(filepath, data)


########################################################################################################################
# define utils for network save and restore
########################################################################################################################

def save_model2json(model, filepath):
    # Save JSON config to filepath with name
    json_config = model.to_json()
    with open(filepath, 'w') as json_file:
        json_file.write(json_config)


def save_model_weights(model, filepath):
    # Save model weights to filepath with name
    model.save_weights(filepath)


def restore_model_from_json(json_path, customize_obejct=None):
    # restore model
    with open(json_path) as json_file:
        json_config = json_file.read()
    model = tf.keras.models.model_from_json(
        json_config, custom_objects=customize_obejct)

    return model


def load_model_weights(model, weights_path):
    # restore model weights from weights_path
    model.load_weights(weights_path)
    return model


def Eucli_dist(x, y):
    a = np.subtract(x, y)
    return np.dot(a.T, a)


def Endmember_extract(x, p):
    [D, N] = x.shape
    # If no distf given, use Euclidean distance function
    Z1 = np.zeros((1, 1))
    O1 = np.ones((1, 1))
    # Find farthest point
    d = np.zeros((p, N))
    I = np.zeros((p, 1))
    V = np.zeros((1, N))
    ZD = np.zeros((D, 1))
    # if nargin<4
    for i in range(N):
        d[0, i] = Eucli_dist(x[:, i].reshape(D, 1), ZD)
    # d[0,i]=l1_distance(x[:,i].reshape(D,1),ZD)
    # else
    #     for i=1:N
    #         d(1,i)=distf(x(:,i),zeros(D,1),opt);

    I = np.argmax(d[0, :])

    # if nargin<4
    for i in range(N):
        d[0, i] = Eucli_dist(x[:, i].reshape(D, 1), x[:, I].reshape(D, 1))
        # d[0,i] = l1_distance(x[:,i].reshape(D,1),x[:,I].reshape(D,1))

    # else
    #     for i=1:N
    #         d(1,i)=distf(x(:,i),x(:,I(1)),opt);
    for v in range(1, p):
        # D=[d[0:v-2,I] ; np.ones((1,v-1)) 0]
        D1 = np.concatenate(
            (d[0:v, I].reshape((v, I.size)), np.ones((v, 1))), axis=1)
        D2 = np.concatenate((np.ones((1, v)), Z1), axis=1)
        D4 = np.concatenate((D1, D2), axis=0)
        D4 = np.linalg.inv(D4)
        for i in range(N):
            D3 = np.concatenate((d[0:v, i].reshape((v, 1)), O1), axis=0)
            V[0, i] = np.dot(np.dot(D3.T, D4), D3)

        I = np.append(I, np.argmax(V))
        # if nargin<4
        for i in range(N):
            # d[v,i]=l1_distance(x[:,i].reshape(D,1),x[:,I[v]].reshape(D,1))
            d[v, i] = Eucli_dist(x[:, i].reshape(
                D, 1), x[:, I[v]].reshape(D, 1))

        # else
        #     for i=1:N
        #         d(v,i)=distf(x(:,i),x(:,I(v)),opt);
    per = np.argsort(I)
    I = np.sort(I)
    d = d[per, :]
    return I, d


def Endmember_reorder2(A, E1):
    index = []
    _, p = A.shape
    _, q = E1.shape
    for l in range(p):
        error = np.ones((1, q)) * 10000
        for n in range(q):
            if n not in index:
                error[0, n] = Eucli_dist(A[:, l], E1[:, n])
        b = np.argmin(error)
        index = np.append(index, b)
    index = index.astype(int)
    return index
