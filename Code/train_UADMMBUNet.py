"""
Description:
In this script, we train and evaluate the Unfolding ADMM network for hyperspectral image unmixing, UADMMBUNet in
https://ieeexplore.ieee.org/abstract/document/9654204
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

import scipy
import tensorflow as tf
from timeit import default_timer as timer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from Modules.utils import *
from Modules.Model import *
from Modules.Losses import *
from Modules.Synthetic_data_preprocessing import *
from Modules.VCA import *



__author__ = '{Chao ZHOU}'
__copyright__ = 'Copyright {04/09/2020}, {UADMMAENet}'
__email__ = '{chaozhouucl@gmail.com}'
__status__ = '{finished}'


# gpu setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


# {code}
def main(hparams):
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # ------------------------------------------  1. load data ------------------------------------------
    data_params = {
        'HSI data path':  os.path.join(parent_dir, 'Data/hsi_data_Sigma1.414.pkl'),
        'True abundance path': os.path.join(parent_dir, 'Data/abundance_Sigma1.414.pkl'),
        'True Endmember signature path': os.path.join(parent_dir, 'Data/endm_sigs_Sigma1.414.pkl'),
        'img_size': (100, 100),
    }

    hsi_data = read_data(data_params['HSI data path'])
    true_abundances = read_data(data_params['True abundance path'])
    true_endm_sig = read_data(data_params['True Endmember signature path'])

    NO_DATA, NO_Bands = hsi_data.shape
    _, NO_Endms = true_abundances.shape

    data_params['NO_Bands'] = NO_Bands
    data_params['NO_Endms'] = NO_Endms
    data_params['NO_DATA'] = NO_DATA

    assert NO_DATA == np.prod(data_params['img_size'])

    # add noise
    hsi_data = add_gaussian_noise(hsi_data, hparams['SNR'])

    # clip abundance to avoid overflow of loss and metric calculation
    true_abundances = np.clip(true_abundances, 1e-8, 0.99)

    train_x = hsi_data
    train_y = hsi_data
    train_pseudo_z0 = np.zeros_like(true_abundances)
    train_pseudo_d0 = np.zeros_like(true_abundances)
    train_x = [train_x, train_pseudo_z0, train_pseudo_d0]

    if hparams['endmember_estimation_method'] == 'SiVM':
        img_resh = hsi_data.T
        V, SS, U = scipy.linalg.svd(img_resh, full_matrices=False)
        PC = np.diag(SS) @ U
        img_resh_DN = V[:, :NO_Endms] @ PC[:NO_Endms, :]
        img_resh_np_clip = np.clip(img_resh_DN, 0, 1)
        II, III = Endmember_extract(img_resh_np_clip, NO_Endms)
        E_np1 = img_resh_np_clip[:, II]
        asq = Endmember_reorder2(true_endm_sig, E_np1)
        sivm_endm = E_np1[:, asq]
        Endm_initialization = sivm_endm

    elif hparams['endmember_estimation_method'] == 'VCA':
        vca_est_endm, _, _ = vca(hsi_data.T, R=data_params['NO_Endms'])
        Endm_initialization = np.abs(vca_est_endm)
    

    # ------------------------------------------ 2. build model ------------------------------------------
    model_params = {
        'input_shape': (NO_Bands,),
        'output_shape': (NO_Endms,),
        'number_layers': hparams['number_layers'],
        'share_layers': True if hparams['network_type'] == 1 else False,
        'A': Endm_initialization,  # encoder initialization
        'Endm_initialization': Endm_initialization,  # decoder initialization
        'lambda_0': 0.1,
        'name': 'UADMMBUNet-I' if hparams['network_type'] == 1 else 'UADMMBUNet-II',

    }

    # UADMMBUNet customize objects
    UADMMBUNet_customize_obejct = {
        'UADMMNet_Reconstruction_layer': UADMMNet_Reconstruction_layer,
        'UADMMNet_AuxiliaryVariableUpdate_layer': UADMMNet_AuxiliaryVariableUpdate_layer,
        'UADMMNet_Multiplier_layer': UADMMNet_Multiplier_layer,
        'UADMMNet_Norm_layer': UADMMNet_Norm_layer,
        'MinMaxVal': MinMaxVal

    }

    model = UADMM_BUNet(**model_params).model
    optimizer = optimizers.Adam(learning_rate=hparams['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='mse',
    )

    # ---------------------------- 3. experiment log ----------------------------
    Readme = f"evaluate model: {model_params['name']} on synthetic dataset.\r\n"\
        f"Hyperparameters: {hparams}.\r\n"

    kwargs = {
        'Readme': Readme,

    }
    if not os.path.exists(os.path.join(parent_dir, 'Result/')):
        os.mkdir(os.path.join(parent_dir, 'Result/'))
    program_log_path, model_checkpoint_dir, tensorboard_log_dir, model_log_dir = create_project_log_path(
        project_path=os.path.join(parent_dir, 'Result/'), **kwargs)

    # Save JSON config to disk
    save_model2json(model,
                    model_checkpoint_dir + model_params['name'] + '_model_config.json')
    model.save_weights(model_checkpoint_dir + model_params['name'])

    # summary model to Readme
    summary_model2_readme(model, readme_path=program_log_path + 'Readme.txt')

    # ---------------------------- 4. train ----------------------------
    callbacks = [
        ModelCheckpoint(
            filepath=model_checkpoint_dir +
            model_params['name'],  # + '_weights.h5',
            save_best_only=True,
            save_weights_only=True,
            monitor='loss'),
    ]
    start = timer()
    model.fit(x=train_x, y=train_y,
              callbacks=callbacks,
              batch_size=hparams['batch_size'],
              epochs=hparams['epochs'],
              verbose=hparams['verbose']
              )
    train_time = timer() - start
    # free memory
    del model
    # ---------------------------- 5. evaluate ----------------------------
    # load model
    new_model = restore_model_from_json(
        model_checkpoint_dir + model_params['name'] + '_model_config.json',
        customize_obejct=UADMMBUNet_customize_obejct
    )
    new_model = load_model_weights(
        new_model,
        model_checkpoint_dir + model_params['name'],
    )

    # extract endms
    est_endm_sig = new_model.layers[-1].get_weights()[0].T
    asq = Endmember_reorder2(true_endm_sig, est_endm_sig)
    est_endm_sig = est_endm_sig[:, asq]

    # est arbundance
    abu = new_model.layers[-2].output
    abu_net = Model(new_model.inputs, abu)

    # evaluation
    start = timer()
    est_abundance = abu_net.predict(x=train_x,
                                    batch_size=hparams['batch_size'],
                                    )
    eval_time = timer() - start
    est_abundance = est_abundance[:, asq]

    # free memory
    del new_model, abu_net

    write_data(est_endm_sig, file_path=tensorboard_log_dir + 'est_endm_sig.pkl')
    write_data(est_abundance, file_path=tensorboard_log_dir +
               'est_abundance.pkl')

    # ---------------------------- post_processing ----------------------------
    # calc metrics for initialization method
    sad = angle_distance_metric(true_endm_sig.T, Endm_initialization.T)

    # print and summary
    summary_str = hparams['endmember_estimation_method'] + \
        f" est Endms SAD: {sad}.\r\n"
    print(summary_str)
    summary2readme(summary_str, readme_path=program_log_path + 'Readme.txt')

    # calc metrics for UADMM-BUNet endm_sig
    sad = angle_distance_metric(true_endm_sig.T, est_endm_sig.T)

    # print and summary
    summary_str = model_params['name'] + f" est Endms SAD: {sad}.\r\n"
    print(summary_str)
    summary2readme(summary_str, readme_path=program_log_path + 'Readme.txt')

    # calc abundance metrics
    rmse = RMSE_metric(true_abundances, est_abundance)
    aad = angle_distance_metric(true_abundances, est_abundance)
    aid = abundance_information_divergence_metric(
        true_abundances, est_abundance)

    # print and summary RMSE
    summary_str = model_params['name'] + ' est Abundance:\r\n' \
        + 'RMSE: %f\r\n' % (rmse) \
        + 'AAD: %f\r\n' % (aad) \
        + 'AID: %f\r\n' % (aid) \
        + 'Eval time: %f s\r\n' % (eval_time) \
        + 'Train time: %f s\r\n' % (train_time)
    print(summary_str)
    summary2readme(summary_str, readme_path=program_log_path + 'Readme.txt')


if __name__ == '__main__':
    # ----------------------------------------------------------------hyper-parameters----------------------------------------------------------------
    hparams = {
        'verbose': 0,
        'SNR': 25,
        'network_type': 2,
        'endmember_estimation_method': 'VCA',
        'number_layers': 2,
        'learning_rate': 1e-4,
        'epochs': 300,
        'batch_size': 64,
    }
    # ----------------------------------------------------------------main----------------------------------------------------------------
    main(hparams)
