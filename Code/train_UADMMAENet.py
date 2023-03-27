'''
Description:
In this script, we train and evaluate the Unfolding ADMM network for hyperspectral image unmixing, UADMMAENet in
https://ieeexplore.ieee.org/abstract/document/9654204
'''

# Futures
from __future__ import print_function

import sys
import os

# getting the directory where this file is located
current_dir = os.path.dirname(os.path.realpath(__file__))
# getting the parent directory and adding it to the path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Modules.Synthetic_data_preprocessing import *
from Modules.Losses import *
from Modules.Model import *
from Modules.utils import *
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from timeit import default_timer as timer
import tensorflow as tf



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

    # split into training data and testing data
    train_index, eval_index = sampling(
        NO_DATA=NO_DATA, sample_size=hparams['train_size'])

    # train data
    train_x = hsi_data[train_index, :]
    train_y = true_abundances[train_index, :]
    train_pseudo_z0 = np.zeros_like(train_y)
    train_pseudo_d0 = np.zeros_like(train_y)
    train_x = [train_x,
               train_pseudo_z0, train_pseudo_d0]

    # test data
    evaluate_x = hsi_data[eval_index, :]
    evaluate_y = true_abundances[eval_index, :]
    evaluate_pseudo_z0 = np.zeros_like(evaluate_y)
    evaluate_pseudo_d0 = np.zeros_like(evaluate_y)
    evaluate_x = [
        evaluate_x, evaluate_pseudo_z0, evaluate_pseudo_d0]

    # ------------------------------------------ 2. build model ------------------------------------------
    # model params
    model_params = {
        'input_shape': (NO_Bands,),
        'output_shape': (NO_Endms,),
        'number_layers': hparams['number_layers'],
        'share_layers': True if hparams['network_type'] == 1 else False,
        'A': np.copy(true_endm_sig),
        'lambda_0': 0.1,
        'name': 'UADMMAENet-I' if hparams['network_type'] == 1 else 'UADMMAENet-II',
    }

    # UADMMUNet customize objects
    UADMMUnet_customize_obejct = {
        'UADMMNet_Reconstruction_layer': UADMMNet_Reconstruction_layer,
        'UADMMNet_AuxiliaryVariableUpdate_layer': UADMMNet_AuxiliaryVariableUpdate_layer,
        'UADMMNet_Multiplier_layer': UADMMNet_Multiplier_layer,
        'UADMMNet_Norm_layer': UADMMNet_Norm_layer,
    }

    model = UADMM_AENet(**model_params).model
    optimizer = optimizers.Adam(learning_rate=hparams['learning_rate'])
    loss = customize_loss([hparams['mse_loss_weight'],
                          hparams['aad_loss_weight'], hparams['aid_loss_weight']])
    model.compile(
        optimizer=optimizer,
        loss=loss,
    )

    # ---------------------------- 3. experiment log ----------------------------
    Readme = f"evaluate model: {model_params['name']} on synthetic dataset.\r\n"\
        f"Hyperparameters: {hparams}.\r\n"

    kwargs = {
        'Readme': Readme,
    }

    if not os.path.exists(os.path.join(parent_dir,'Result/')):
        os.mkdir(os.path.join(parent_dir,'Result/'))
    program_log_path, model_checkpoint_dir, tensorboard_log_dir, model_log_dir = create_project_log_path(
        project_path=os.path.join(parent_dir,'Result/'), **kwargs)

    # Save JSON config to disk
    save_model2json(model,
                    model_checkpoint_dir + model_params['name'] + '_model_config.json')
    model.save_weights(model_checkpoint_dir + model_params['name'])

    # summary model to Readme
    summary_model2_readme(model, readme_path=program_log_path + 'Readme.txt')

    # ---------------------------- 4. train ----------------------------
    callbacks = [
        ModelCheckpoint(
            filepath=model_checkpoint_dir + model_params['name'],
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
        model_checkpoint_dir +
        model_params['name'] + '_model_config.json',
        customize_obejct=UADMMUnet_customize_obejct
    )
    new_model = load_model_weights(
        new_model,
        model_checkpoint_dir + model_params['name'],
    )

    # evaluation
    start = timer()
    est_abundance = new_model.predict(x=evaluate_x,
                                      batch_size=hparams['batch_size'],
                                      )
    eval_time = timer() - start

    # free memory
    del new_model

    write_data(est_abundance, file_path=tensorboard_log_dir +
               'est_abundance.pkl')

    # ---------------------------- 6. post_processing ----------------------------
    # calc metrics
    rmse = RMSE_metric(evaluate_y, est_abundance)
    mae = MAE_metric(evaluate_y, est_abundance)
    aad = angle_distance_metric(evaluate_y, est_abundance)
    aid = abundance_information_divergence_metric(evaluate_y, est_abundance)

    # print and summary RMSE
    summary_str = model_params['name'] + 'est abundance:\r\n' \
        + 'Train time: %f s\r\n' % (train_time) \
        + 'Eval time: %f s\r\n' % (eval_time) \
        + 'RMSE: %f\r\n' % (rmse) \
        + 'MAE: %f\r\n' % (mae) \
        + 'AAD: %f\r\n' % (aad) \
        + 'AID: %f\r\n' % (aid)
    print(summary_str)

    summary2readme(summary_str, readme_path=program_log_path + 'Readme.txt')


if __name__ == '__main__':
    # ----------------------------------------------------------------hyper-parameters----------------------------------------------------------------
    hparams = {
        'verbose': 1,
        'batch_size': 64,
        'epochs': 300,
        'number_layers': 2,
        'train_size': 1000,
        'SNR': 15,
        'network_type': 2,
        'learning_rate': 1e-4,
        'mse_loss_weight': 1.,
        'aad_loss_weight': 1e-7,
        'aid_loss_weight': 1e-5,
    }
    # ----------------------------------------------------------------main----------------------------------------------------------------

    main(hparams)
